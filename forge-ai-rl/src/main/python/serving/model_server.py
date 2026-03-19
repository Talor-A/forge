"""
Model Server — TCP server that receives inference requests from Java and returns decisions.

Protocol: length-prefixed JSON over TCP
- Client sends: [4 bytes big-endian length][JSON payload]
- Server responds: [4 bytes big-endian length][JSON payload]

This is the bridge between the Java game engine and the Python neural network.
"""

import json
import socket
import struct
import threading
import logging
import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mtg_model import MTGModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ModelServer:
    """
    TCP server for serving RL model inference to the Java game engine.
    """

    def __init__(self, model: MTGModel, host: str = 'localhost', port: int = 50051,
                 device: str = 'cpu'):
        self.model = model
        self.model.eval()
        self.host = host
        self.port = port
        self.device = device
        self.running = False
        self.request_count = 0
        # Lock to serialize CUDA inference (not thread-safe)
        self._inference_lock = threading.Lock()
        # Limit concurrent client threads to prevent OOM
        self._client_semaphore = threading.Semaphore(32)

    def start(self):
        """Start the server."""
        self.running = True
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(128)
        logger.info(f"Model server listening on {self.host}:{self.port}")

        try:
            while self.running:
                server_socket.settimeout(1.0)
                try:
                    client_socket, addr = server_socket.accept()
                    thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, addr))
                    thread.daemon = True
                    thread.start()
                except socket.timeout:
                    continue
                except OSError:
                    continue
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.running = False
            try:
                server_socket.close()
            except Exception:
                pass

    def _handle_client(self, client_socket: socket.socket, addr):
        """Handle a single client connection."""
        self._client_semaphore.acquire()
        try:
            client_socket.settimeout(300)  # 5 min timeout
            while self.running:
                # Read length prefix
                length_bytes = self._recv_exact(client_socket, 4)
                if not length_bytes:
                    break
                length = struct.unpack('>I', length_bytes)[0]
                if length > 10_000_000:  # 10MB sanity limit
                    break

                # Read payload
                payload = self._recv_exact(client_socket, length)
                if not payload:
                    break

                # Parse and process
                try:
                    request = json.loads(payload.decode('utf-8'))
                    self.request_count += 1
                    response = self._process_request(request)
                except Exception as e:
                    response = {
                        'selectedIndices': [0],
                        'actionProbabilities': [],
                        'valueEstimate': 0.0,
                    }

                # Send response
                response_bytes = json.dumps(response).encode('utf-8')
                client_socket.sendall(struct.pack('>I', len(response_bytes)))
                client_socket.sendall(response_bytes)

        except Exception:
            pass  # Client disconnected or error
        finally:
            self._client_semaphore.release()
            try:
                client_socket.close()
            except Exception:
                pass

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes."""
        data = b''
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    @torch.no_grad()
    def _process_request(self, request: dict) -> dict:
        """Process an inference request and return the decision.
        Serialized via lock — CUDA is not thread-safe."""
        try:
            with self._inference_lock:
                return self._process_request_impl(request)
        except Exception as e:
            return {
                'selectedIndices': [0],
                'actionProbabilities': [],
                'valueEstimate': 0.0,
            }

    def _process_request_impl(self, request: dict) -> dict:
        decision_type = request.get('decisionType', 'UNKNOWN')

        try:
            # Parse game state features
            state_embedding = self._encode_game_state(request)

            if decision_type == 'PRIORITY_ACTION':
                return self._handle_priority(state_embedding, request)
            elif decision_type == 'TARGET_SELECTION':
                return self._handle_target(state_embedding, request)
            elif decision_type == 'DECLARE_ATTACKERS':
                return self._handle_attack(state_embedding, request)
            elif decision_type == 'DECLARE_BLOCKERS':
                return self._handle_block(state_embedding, request)
            elif decision_type == 'CARD_SELECTION':
                return self._handle_card_select(state_embedding, request)
            elif decision_type == 'MULLIGAN':
                return self._handle_mulligan(state_embedding, request)
            elif decision_type == 'BINARY_CHOICE':
                return self._handle_binary(state_embedding, request)
            else:
                # Default: return first option
                return {'selectedIndices': [0], 'actionProbabilities': [], 'valueEstimate': 0.0}

        except Exception as e:
            logger.error(f"Error processing {decision_type}: {e}")
            return {'selectedIndices': [0], 'actionProbabilities': [], 'valueEstimate': 0.0}

    def _encode_game_state(self, request: dict) -> torch.Tensor:
        """Encode game state features into a state embedding."""
        global_features = self._to_tensor(request.get('globalFeatures', []), (1, 64))

        my_board = self._to_tensor_2d(request.get('myBoardFeatures', []), 30, 128)
        my_board_mask = self._to_mask(request.get('myBoardMask', []), 30)
        opp_board = self._to_tensor_2d(request.get('oppBoardFeatures', []), 30, 128)
        opp_board_mask = self._to_mask(request.get('oppBoardMask', []), 30)
        hand = self._to_tensor_2d(request.get('myHandFeatures', []), 15, 128)
        hand_mask = self._to_mask(request.get('myHandMask', []), 15)

        # Use zero tensors for graveyard/stack to keep it simple initially
        gy_size, stack_size = 40, 10
        my_gy = torch.zeros(1, gy_size, 128, device=self.device)
        my_gy_mask = torch.zeros(1, gy_size, dtype=torch.bool, device=self.device)
        opp_gy = torch.zeros(1, gy_size, 128, device=self.device)
        opp_gy_mask = torch.zeros(1, gy_size, dtype=torch.bool, device=self.device)
        stack = self._to_tensor_2d(request.get('stackFeatures', []), stack_size, 128)
        stack_mask = self._to_mask(request.get('stackMask', []), stack_size)

        return self.model.encode_state(
            global_features, my_board, my_board_mask, opp_board, opp_board_mask,
            hand, hand_mask, my_gy, my_gy_mask, opp_gy, opp_gy_mask, stack, stack_mask
        )

    def _handle_priority(self, state: torch.Tensor, request: dict) -> dict:
        candidates = request.get('candidateFeatures', [])
        if not candidates:
            return {'selectedIndices': [0], 'actionProbabilities': [], 'valueEstimate': 0.0}

        action_features = self._to_tensor_2d(candidates, len(candidates), 64)
        action_mask = torch.ones(1, len(candidates), dtype=torch.bool, device=self.device)

        logits = self.model.priority_head(state, action_features, action_mask)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        value = self.model.get_value(state).item()

        return {
            'selectedIndices': [action],
            'actionProbabilities': probs[0].cpu().numpy().tolist(),
            'valueEstimate': value,
        }

    def _handle_target(self, state: torch.Tensor, request: dict) -> dict:
        candidates = request.get('candidateFeatures', [])
        if not candidates:
            return {'selectedIndices': [0], 'actionProbabilities': [], 'valueEstimate': 0.0}

        target_features = self._to_tensor_2d(candidates, len(candidates), 64)
        target_mask = torch.ones(1, len(candidates), dtype=torch.bool, device=self.device)

        logits = self.model.target_head(state, target_features, target_mask)
        probs = torch.softmax(logits, dim=-1)

        max_select = request.get('maxSelections', 1)
        if max_select <= 1:
            action = torch.multinomial(probs, 1).item()
            selected = [action]
        else:
            # Select top-k by probability
            _, indices = probs.topk(min(max_select, len(candidates)), dim=-1)
            selected = indices[0].cpu().numpy().tolist()

        value = self.model.get_value(state).item()
        return {
            'selectedIndices': selected,
            'actionProbabilities': probs[0].cpu().numpy().tolist(),
            'valueEstimate': value,
        }

    def _handle_attack(self, state: torch.Tensor, request: dict) -> dict:
        candidates = request.get('candidateFeatures', [])
        if not candidates:
            return {'selectedIndices': [], 'actionProbabilities': [], 'valueEstimate': 0.0}

        creature_features = self._to_tensor_2d(candidates, len(candidates), 128)
        creature_mask = torch.ones(1, len(candidates), dtype=torch.bool, device=self.device)

        logits = self.model.attack_head(state, creature_features, creature_mask)
        probs = torch.sigmoid(logits)
        decisions = torch.bernoulli(probs.clamp(0, 1))

        selected = [i for i in range(len(candidates)) if decisions[0, i].item() > 0.5]
        value = self.model.get_value(state).item()

        return {
            'selectedIndices': selected,
            'actionProbabilities': probs[0].cpu().numpy().tolist(),
            'valueEstimate': value,
        }

    def _handle_block(self, state: torch.Tensor, request: dict) -> dict:
        # Simplified: treat as card selection
        return self._handle_card_select(state, request)

    def _handle_card_select(self, state: torch.Tensor, request: dict) -> dict:
        candidates = request.get('candidateFeatures', [])
        if not candidates:
            return {'selectedIndices': [], 'actionProbabilities': [], 'valueEstimate': 0.0}

        card_features = self._to_tensor_2d(candidates, len(candidates), 128)
        card_mask = torch.ones(1, len(candidates), dtype=torch.bool, device=self.device)

        logits = self.model.card_select_head(state, card_features, card_mask)
        probs = torch.softmax(logits, dim=-1)

        min_select = request.get('minSelections', 1)
        max_select = request.get('maxSelections', 1)
        num_select = max(min_select, min(max_select, len(candidates)))

        if num_select <= 0:
            selected = []
        elif num_select == 1:
            action = torch.multinomial(probs, 1).item()
            selected = [action]
        else:
            _, indices = probs.topk(min(num_select, len(candidates)), dim=-1)
            selected = indices[0].cpu().numpy().tolist()

        value = self.model.get_value(state).item()
        return {
            'selectedIndices': selected,
            'actionProbabilities': probs[0].cpu().numpy().tolist(),
            'valueEstimate': value,
        }

    def _handle_mulligan(self, state: torch.Tensor, request: dict) -> dict:
        candidates = request.get('candidateFeatures', [])
        if not candidates:
            return {'selectedIndices': [1], 'actionProbabilities': [], 'valueEstimate': 0.0}

        hand_features = self._to_tensor_2d(candidates, len(candidates), 128)
        hand_mask = torch.ones(1, len(candidates), dtype=torch.bool, device=self.device)

        keep_logit, _ = self.model.mulligan_head.evaluate_hand(state, hand_features, hand_mask)
        keep_prob = torch.sigmoid(keep_logit).item()
        keep = keep_prob > 0.5

        value = self.model.get_value(state).item()
        return {
            'selectedIndices': [1 if keep else 0],
            'actionProbabilities': [1 - keep_prob, keep_prob],
            'valueEstimate': value,
        }

    def _handle_binary(self, state: torch.Tensor, request: dict) -> dict:
        logit = self.model.binary_head(state)
        prob = torch.sigmoid(logit).item()
        decision = prob > 0.5

        value = self.model.get_value(state).item()
        return {
            'selectedIndices': [1 if decision else 0],
            'actionProbabilities': [1 - prob, prob],
            'valueEstimate': value,
        }

    # Tensor conversion helpers

    def _to_tensor(self, data, target_shape) -> torch.Tensor:
        """Convert a list to a tensor with target shape, padding with zeros."""
        if isinstance(data, list) and len(data) > 0:
            t = torch.tensor(data, dtype=torch.float32, device=self.device)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            # Pad or truncate to target shape
            result = torch.zeros(target_shape, device=self.device)
            min_len = min(t.shape[-1], target_shape[-1])
            result[..., :min_len] = t[..., :min_len]
            return result
        return torch.zeros(target_shape, device=self.device)

    def _to_tensor_2d(self, data, max_items, feature_dim) -> torch.Tensor:
        """Convert a 2D list to a padded tensor."""
        result = torch.zeros(1, max_items, feature_dim, device=self.device)
        if isinstance(data, list):
            for i, row in enumerate(data):
                if i >= max_items:
                    break
                if isinstance(row, list) and len(row) > 0:
                    min_len = min(len(row), feature_dim)
                    result[0, i, :min_len] = torch.tensor(row[:min_len], dtype=torch.float32, device=self.device)
        return result

    def _to_mask(self, data, max_items) -> torch.Tensor:
        """Convert a boolean list to a mask tensor."""
        result = torch.zeros(1, max_items, dtype=torch.bool, device=self.device)
        if isinstance(data, list):
            for i, val in enumerate(data):
                if i >= max_items:
                    break
                result[0, i] = bool(val)
        return result


def main():
    """Main entry point for the model server."""
    import argparse
    parser = argparse.ArgumentParser(description='MTG RL Model Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=50051, help='Server port')
    parser.add_argument('--model', default=None, help='Path to saved model weights')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    args = parser.parse_args()

    # Create or load model
    if args.model and os.path.exists(args.model):
        logger.info(f"Loading model from {args.model}")
        model = MTGModel.load(args.model, device=args.device)
    else:
        logger.info("Creating fresh model with random weights")
        model = MTGModel()
        model.to(args.device)

    # Print parameter counts
    counts = model.count_parameters()
    logger.info(f"Model parameters: {counts}")

    # Start server
    server = ModelServer(model, host=args.host, port=args.port, device=args.device)
    server.start()


if __name__ == '__main__':
    main()

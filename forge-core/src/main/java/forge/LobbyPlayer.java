package forge;

import org.apache.commons.lang3.StringUtils;

import java.util.Objects;

/** 
 * This means a player's part unchanged for all games.
 * 
 * May store player's assets here.
 *
 */
public abstract class LobbyPlayer {
    protected String name;
    private int avatarIndex = -1;
    private int sleeveIndex = -1;
    private String avatarCardImageKey;

    public LobbyPlayer(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
    public void setName(String name0) {
        if (StringUtils.isEmpty(name0)) { return; } //don't allow setting name to nothing
        name = name0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(name);
    }

    /*
     * Two LobbyPlayers are equal if they have the same name
     * and are the same kind of player (human vs AI).
     * Uses instanceof instead of strict class equality so
     * subclasses (e.g. RecordingLobbyPlayerAi) work correctly.
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof LobbyPlayer)) {
            return false;
        }
        LobbyPlayer other = (LobbyPlayer) obj;
        return Objects.equals(name, other.name);
    }

    public int getAvatarIndex() {
        return avatarIndex;
    }
    public int getSleeveIndex() {
        return sleeveIndex;
    }
    public void setAvatarIndex(int avatarIndex) {
        this.avatarIndex = avatarIndex;
    }
    public void setSleeveIndex(int sleeveIndex) {
        this.sleeveIndex = sleeveIndex;
    }

    public String getAvatarCardImageKey() {
        return avatarCardImageKey;
    }
    public void setAvatarCardImageKey(String avatarImageKey0) {
        this.avatarCardImageKey = avatarImageKey0;
    }

    public abstract void hear(LobbyPlayer player, String message);
}

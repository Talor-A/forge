#!/bin/bash
# Step 1: Build the Java fat jar
set -e
cd /home/maustin/forge
echo "Building forge jar..."
mvn package -pl forge-gui-desktop -am -Denforcer.skip=true -Dcheckstyle.skip=true -DskipTests -q
echo "Build complete."

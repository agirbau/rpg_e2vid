#!/bin/bash
set -e

# Download relevant data if missing
[ -f pretrained/E2VID_lightweight.pth.tar ] || \
    wget -c http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar -O pretrained/E2VID_lightweight.pth.tar

[ -f data/dynamic_6dof.zip ] || \
    wget -c http://rpg.ifi.uzh.ch/data/E2VID/datasets/ECD_IJRR17/dynamic_6dof.zip -O data/dynamic_6dof.zip

# Build docker image using Compose
docker-compose build
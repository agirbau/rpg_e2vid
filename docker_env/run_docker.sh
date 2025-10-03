# Not used anymore, see docker-compose.yaml
#!/bin/bash

E2VID_PATH=/home/andreu/work/projects/research/rpg_e2vid
E2VID_DATA=/home/andreu/datasets

# Add your Docker commands here
docker rm e2vid
docker run \
    --name e2vid \
    --gpus '"device=1"' \
    -it \
    -v $E2VID_PATH:/home/user/app \
    -v $E2VID_DATA:/home/user/datasets \
    -e E2VID_PATH=/home/user/app \
    -e E2VID_DATA=/home/user/datasets \
    e2vid
#!/bin/bash

# Build docker image
docker build --no-cache -t e2vid:latest -f docker_env/Dockerfile .
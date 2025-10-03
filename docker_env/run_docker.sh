#!/bin/bash
set -e

# Stop any running containers
docker-compose down

# Start an interactive shell in A
docker-compose run --rm e2vid bash
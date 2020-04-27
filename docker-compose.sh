#!/bin/bash

echo "Once the containers are running you may browse to http://127.0.0.1:7100/ui/"
read -n1 -r -p "Press any key to continue..." key

# Build the images
cd ..
docker build -t praekeltcom/tropical-app:latest .

docker-compose -f docker-compose.yml up
docker-compose -f docker-compose.yml down

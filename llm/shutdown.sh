#!/bin/bash

# This script is called when the LLM container becomes unhealthy
# It ensures all containers and networks are gracefully stopped

echo "LLM container is unhealthy, initiating graceful shutdown of all containers..."

# Get the container ID of the current container
CONTAINER_ID=$(hostname)

# Get the project name from the container ID
PROJECT_NAME=$(echo $CONTAINER_ID | cut -d'-' -f1)

# Stop all containers in the project gracefully
echo "Stopping all containers in project $PROJECT_NAME..."
docker-compose -p $PROJECT_NAME down --remove-orphans

echo "All containers have been stopped gracefully." 
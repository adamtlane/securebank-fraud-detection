#!/bin/bash

IMAGE_NAME="securebank"
DOCKERFILE_PATH="Dockerfile"
DEFAULT_PORT=5000

usage() {
  echo "Usage: $0 [-p port]"
  echo "  -p    Port number to expose and map (default: $DEFAULT_PORT)"
  exit 1
}

while getopts "p:" opt; do
  case $opt in
    p) PORT="$OPTARG" ;;
    *) usage ;;
  esac
done

# Default to port from Dockerfile or fallback
if [ -z "$PORT" ]; then
  PORT=$(grep -i "^EXPOSE" "$DOCKERFILE_PATH" | awk '{print $2}' | tr -d '\r\n')
  if [ -z "$PORT" ]; then
    PORT=$DEFAULT_PORT
  fi
fi

echo "Building Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" .
if [ $? -ne 0 ]; then
  echo "Error: Failed to build Docker image."
  exit 1
fi
echo "Docker image '$IMAGE_NAME' built successfully."
echo "Using PORT: $PORT"

# Ensure host-side folders exist (do NOT overwrite data_sources)
mkdir -p securebank/data_sources
mkdir -p securebank/storage/models securebank/storage/datasets securebank/logs

# Stop and remove existing container with same name
CONTAINER_NAME="securebank_container"
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
  echo "Stopping and removing existing container '$CONTAINER_NAME'..."
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1
  docker rm "$CONTAINER_NAME" >/dev/null 2>&1
fi

echo "Running Docker container '$CONTAINER_NAME' on port $PORT..."
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$PORT:5000" \
  -v "$(pwd)/securebank/data_sources:/app/securebank/data_sources" \
  -v "$(pwd)/securebank/storage:/app/securebank/storage" \
  -v "$(pwd)/securebank/logs:/app/securebank/logs" \
  "$IMAGE_NAME"
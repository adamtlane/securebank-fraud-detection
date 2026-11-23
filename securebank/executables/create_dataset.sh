#!/bin/bash

HOST="localhost"
PORT="5000"
ENDPOINT="create_dataset"

usage() {
  echo "Usage: $0 [-h host] [-p port]"
  echo "  -h    Hostname or IP of the API server (default: localhost)"
  echo "  -p    Port number of the API server (default: 5000)"
  exit 1
}

while getopts "h:p:" opt; do
  case $opt in
    h) HOST="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    *) usage ;;
  esac
done

# Warn if host-side data_sources is empty
if [ ! -d "securebank/data_sources" ] || [ -z "$(ls -A securebank/data_sources)" ]; then
  echo "Warning: 'securebank/data_sources' is empty. Ensure your raw data files are present."
fi

URL="http://$HOST:$PORT/$ENDPOINT"

echo "Sending POST request to $URL ..."
response=$(curl -s -w "\nHTTP_STATUS_CODE:%{http_code}" -X POST "$URL")

body=$(echo "$response" | sed -e 's/HTTP_STATUS_CODE\:.*//g')
status=$(echo "$response" | tr -d '\n' | sed -e 's/.*HTTP_STATUS_CODE://')

echo "Response status: $status"
echo "Response body:"
echo "$body"

if [ "$status" -ne 200 ]; then
  echo "Error: /create_dataset request failed."
  exit 1
fi

echo "Dataset creation triggered successfully."
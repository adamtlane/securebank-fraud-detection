#!/bin/bash

HOST="localhost"
PORT="5000"
JSON_FILE="securebank/storage/test.json"

usage() {
  echo "Usage: $0 [-h host] [-p port] [-f json_file]"
  echo "  -h    Hostname or IP of the API server (default: localhost)"
  echo "  -p    Port number of the API server (default: 5000)"
  echo "  -f    Path to JSON file with transaction data (default: securebank/storage/test.json)"
  exit 1
}

while getopts "h:p:f:" opt; do
  case $opt in
    h) HOST="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    f) JSON_FILE="$OPTARG" ;;
    *) usage ;;
  esac
done

if [ ! -f "$JSON_FILE" ]; then
  echo "Error: JSON file '$JSON_FILE' not found."
  echo "Please place your test JSON file inside 'securebank/storage/' or specify a path with -f."
  exit 1
fi

URL="http://$HOST:$PORT/predict/"

echo "Sending POST request to $URL with data from $JSON_FILE..."

response=$(curl -s -w "\nHTTP_STATUS_CODE:%{http_code}" -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d @"$JSON_FILE")

body=$(echo "$response" | sed -e 's/HTTP_STATUS_CODE\:.*//g')
status=$(echo "$response" | tr -d '\n' | sed -e 's/.*HTTP_STATUS_CODE://')

echo "Response status: $status"
echo "Response body:"
echo "$body"

if [ "$status" -ne 200 ]; then
  echo "Error: /predict request failed."
  exit 1
fi

echo "Prediction request succeeded."
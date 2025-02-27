#!/bin/bash
# Script to run the patient app with authentication enabled

# Path to your studies directory
STUDIES_DIR=${1:-"/path/to/your/studies"}

# Set environment variables for authentication
export MONAI_LABEL_AUTH_ENABLE=true
# For basic authentication
export MONAI_LABEL_AUTH_CLIENT_ID=monailabel-client
export MONAI_LABEL_AUTH_USERNAME=admin
export MONAI_LABEL_AUTH_PASSWORD=password123

# Start the server with the patient app
monailabel start_server \
  --app /home/stijn/code/MONAILabel/sample-apps/patient \
  --studies ${STUDIES_DIR} \
  --conf models segmentation,sam

echo "Server started with authentication enabled"

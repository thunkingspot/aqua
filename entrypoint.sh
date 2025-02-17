#!/bin/bash

# Validate Nginx configuration
nginx -t

# Check if the Nginx configuration is valid
if [ $? -ne 0 ]; then
  echo "Nginx configuration is invalid. Exiting."
  exit 1
fi

# Start Nginx
service nginx start

# Start Uvicorn
uvicorn app.service.main:app --host 0.0.0.0 --port 8000
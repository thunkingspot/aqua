# This script is transferred to the host and run locally there.
set -e

echo "DEBUG_MODE: $1"
echo "CONTAINER_NAME: $2"
echo "DOCKER_IMAGE: $3"

DEBUG_MODE=false
if [ "$1" == "true" ]; then
  DEBUG_MODE=true
fi

CONTAINER_NAME=$2
DOCKER_IMAGE=$3

# Clean up old containers and images. Don't fail if there are no containers or images to clean up.
# Do this first because it will clean up all but the current version
# leaving it in place for a potential recovery
if [ "$(sudo docker ps -q)" ]; then
  sudo docker stop $(sudo docker ps -q)
  sudo docker container prune -f
fi
sudo docker image prune -f || true

# Setup log directory
if [ ! -d /var/log/aqua_app ]; then
  sudo mkdir -p /var/log/aqua_app
  sudo chown -R ubuntu:ubuntu /var/log/aqua_app
  sudo chmod -R 755 /var/log/aqua_app
fi

# Load the Docker image
sudo docker load -i $DOCKER_IMAGE

# Create a systemd service to autostart the Docker container
echo "[Unit]
Description=Aqua App Docker Container
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker run --rm -d -p 80:80 -p 5678:5678 -e DEBUG_MODE=$DEBUG_MODE $CONTAINER_NAME
ExecStop=/usr/bin/docker stop \$(/usr/bin/docker ps -q --filter ancestor=$CONTAINER_NAME)

[Install]
WantedBy=multi-user.target" | sudo tee /etc/systemd/system/$CONTAINER_NAME.service

# Reload the systemd config and start the Docker container
sudo systemctl daemon-reload
sudo systemctl enable $CONTAINER_NAME.service
sudo systemctl start $CONTAINER_NAME.service

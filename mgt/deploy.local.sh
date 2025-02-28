set -e

# Clean up old containers and images. Don't fail if there are no containers or images to clean up.
# Do this first because it will leave the current version alone
if [ "$(sudo docker ps -q)" ]; then
  sudo docker stop $(sudo docker ps -q)
  sudo docker container prune -f
fi
sudo docker image prune -f || true

if [ "$DEBUG_MODE" == "true" ]; then
  sudo docker run -d -p 80:80 -p 5678:5678 -e DEBUG_MODE=true aqua-app
    else
  sudo docker run -d -p 80:80 -p 5678:5678 -e DEBUG_MODE=false aqua-app
    fi
    
sudo docker ps

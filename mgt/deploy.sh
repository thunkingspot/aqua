set -e

DEBUG_MODE=true
if [ "$1" == "false" ]; then
  DEBUG_MODE=false
fi

sudo docker build -t aqua-app ..

if [ "$DEBUG_MODE" == "true" ]; then
  sudo docker run -d -p 80:80 -p 5678:5678 -e DEBUG_MODE=true aqua-app
    else
  sudo docker run -d -p 80:80 -p 5678:5678 -e DEBUG_MODE=false aqua-app
    fi
sudo docker ps

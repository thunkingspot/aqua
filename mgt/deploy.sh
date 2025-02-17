    DEBUG_MODE=true
    if [ "$1" == "false" ]; then
      DEBUG_MODE=false
    fi

    sudo docker stop $(sudo docker ps -a -q --filter ancestor=aqua-app)
    containers=$(sudo docker ps -a -q --filter ancestor=aqua-app)
    if [ $(echo "$containers" | wc -l) -gt 1 ]; then
      sudo docker rm $(echo "$containers" | tail -n +2)
    fi
    #sudo docker images -q aqua-app | tail -n +2 | xargs sudo docker rmi
    sudo docker build -t aqua-app .

    if [ "$DEBUG_MODE" == "true" ]; then
      sudo docker run -d -p 80:80 -p 5678:5678 -e DEBUG_MODE=true aqua-app
        else
      sudo docker run -d -p 80:80 -p 5678:5678 -e DEBUG_MODE=false aqua-app
        fi
    sudo docker ps

set -e

CONTAINER_NAME=$1
AWS_KEY_PATH=$2
AWS_INSTANCE_USER=$3
AWS_INSTANCE_IP=$4
DEBUG_MODE=false
if [ "$5" == "true" ]; then
  DEBUG_MODE=true
fi

echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "AWS_KEY_PATH: $AWS_KEY_PATH"
echo "AWS_INSTANCE_USER: $AWS_INSTANCE_USER"
echo "AWS_INSTANCE_IP: $AWS_INSTANCE_IP"
echo "DEBUG_MODE: $DEBUG_MODE"

# Transfer the Docker image to the AWS instance
scp -i $AWS_KEY_PATH $CONTAINER_NAME.tar $AWS_INSTANCE_USER@$AWS_INSTANCE_IP:/home/$AWS_INSTANCE_USER/
#green sudo scp -i /home/ubuntu/.ssh/aqua-key2.pem aqua-app.tar ubuntu@10.0.147.201:/home/ubuntu/
#blue sudo scp -i /home/ubuntu/.ssh/aqua-key2.pem aqua-app.tar ubuntu@10.0.138.20:/home/ubuntu/

# Transfer the deploy-remote-container.sh script to the AWS instance
scp -i $AWS_KEY_PATH ./mgt/deploy-remote-container.sh $AWS_INSTANCE_USER@$AWS_INSTANCE_IP:/home/$AWS_INSTANCE_USER/
#green sudo scp -i /home/ubuntu/.ssh/aqua-key2.pem ./mgt/deploy-remote-container.sh ubuntu@10.0.147.201:/home/ubuntu/
#blue sudo scp -i /home/ubuntu/.ssh/aqua-key2.pem ./mgt/deploy-remote-container.sh ubuntu@10.0.138.20:/home/ubuntu/

# Run the deploy-remote-container.sh script on the AWS instance
sudo ssh -i $AWS_KEY_PATH $AWS_INSTANCE_USER@$AWS_INSTANCE_IP << 'EOF'
  /bin/bash /home/$AWS_INSTANCE_USER/deploy-remote-container.sh \
    '"$DEBUG_MODE"' \
    '"$CONTAINER_NAME"' \
    '"$CONTAINER_NAME.tar"'
EOF

set -e

DEBUG_MODE_PARAM=$1
IP_ADDR=$2

DEBUG_MODE=false
if [ "$DEBUG_MODE_PARAM" == "true" ]; then
  DEBUG_MODE=true
fi

# Define AWS instance details
CONTAINER_NAME="aqua-app"
AWS_INSTANCE_USER="ubuntu"
AWS_INSTANCE_IP=$IP_ADDR
AWS_KEY_PATH="/home/ubuntu/.ssh/aqua-key2.pem"

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

set -e

DEBUG_MODE=true
if [ "$1" == "false" ]; then
  DEBUG_MODE=false
fi

# Build the Docker image
sudo docker build -t aqua-app .

# Save the Docker image to a tar file
sudo docker save -o aqua-app.tar aqua-app

# Define AWS instance details
AWS_INSTANCE_USER="ubuntu"
AWS_INSTANCE_IP="10.0.147.201"
AWS_KEY_PATH="/home/ubuntu/.ssh/aqua-key2.pem"

# Transfer the Docker image to the AWS instance
scp -i $AWS_KEY_PATH aqua-app.tar $AWS_INSTANCE_USER@$AWS_INSTANCE_IP:/home/$AWS_INSTANCE_USER/
#scp -i /home/ubuntu/.ssh/aqua-key2.pem aqua-app.tar ubuntu@10.0.147.201:/home/ubuntu/

# Load the Docker image and run the container on the AWS instance
ssh -i $AWS_KEY_PATH $AWS_INSTANCE_USER@$AWS_INSTANCE_IP << EOF
  sudo docker load -i /home/$AWS_INSTANCE_USER/aqua-app.tar
  sudo docker run -d -p 80:80 -p 5678:5678 -e DEBUG_MODE=$DEBUG_MODE aqua-app
EOF

#ssh -i /home/ubuntu/.ssh/aqua-key2.pem ubuntu@10.0.147.201 << EOF
#  sudo docker load -i /home/ubuntu/aqua-app.tar
#  sudo docker run -d -p 80:80 -p 5678:5678 -e DEBUG_MODE=false aqua-app
#EOF

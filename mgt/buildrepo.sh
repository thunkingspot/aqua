set -e

CONTAINER_NAME=$1
REPO_MGT_DIR=$2
TEMP_DIR=$3
STAGE_DIR=$4

# Rememeber current directory.
CURRENT_DIR=$(pwd)
cd $TEMP_DIR

echo "CONTAINER_NAME: $CONTAINER_NAME"
echo "REPO_MGT_DIR: $REPO_MGT_DIR"
echo "TEMP_DIR: $TEMP_DIR"
echo "STAGE_DIR: $STAGE_DIR"
echo "CURRENT_DIR: $CURRENT_DIR"

# Build the Docker image
sudo docker build -t $CONTAINER_NAME .

# Clean up any previous artifacts in the stage directory
if [ -d "$STAGE_DIR" ]; then
  rm -rf $STAGE_DIR
fi
mkdir -p $STAGE_DIR

# Save the Docker image to a tar file
sudo docker save -o $STAGE_DIR/$CONTAINER_NAME.tar $CONTAINER_NAME

# Copy the repo mgt directory to the stage directory
cp -r $REPO_MGT_DIR $STAGE_DIR

cd $CURRENT_DIR
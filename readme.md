- See the mgtsvr repo readme for aws config and build/deploy pipeline info
- deploy.sh requires github ssh key to be present (and configured in github)
- Manual steps to run Aqua app locally
  - Create FastAPI app
    if not already make sure the following are installed globally
      - apt-get update && apt-get install -y nginx libgl1-mesa-glx libglib2.0-0
    Python is installed. But need to install python3.10-venv for virtual env
      sudo apt install python3.10-venv
      from project root create private env
        python3 -m venv venv
        source venv/bin/activate (deactivate to shutdown virtual env)
      with venv active
        pip install -r /app/service/requirements.txt
        pip install pytest httpx
    - install app as a package for testing
      - from app directory (to mark it as a package)
        - touch __init__.py
      - from project root (where setup.py is located)
        - pip install -e .
    - from tests directory
      - pytest
    - use pytest runner
      - needs settings.json in vscode dir
  - Setup website with Nginx
    sudo apt update
    sudo apt install nginx
    - Create web root directory for static files.
      sudo mkdir /var/www/aqua_app
      sudo chown -R www-data:www-data /var/www/aqua_app
      sudo chmod -R 755 /var/www/aqua_app
      NOTE: tried creating a link to the project directory with the website files. There are two problems (at least). The project website directory is no longer owned by the content editor which is a pain. NGINX wasn't happy about it. Did not run it to ground.
    - Modify appinfra/nginx_aqua_local.conf to work with app files.
    - Create a symbolic link to enable the configuration by linking it to the sites-enabled directory.
      sudo ln -s /home/ubuntu/src/aqua/appinfra/nginx_aqua_local.conf /etc/nginx/sites-enabled/
    - Remove the default Nginx configuration link to avoid conflicts.
      sudo rm /etc/nginx/sites-enabled/default
    - Test Nginx configuration to ensure there are no syntax errors.
      sudo nginx -t
    - Reload Nginx to apply the new configuration.
      sudo systemctl reload nginx
    - Nginx start, stop, restart
      sudo systemctl xxx nginx
    - Nginx logs
      sudo tail -f /var/log/nginx/error.log
  - Run the fastapi server - --reload automatically detects changes when in dev cycle
    - run from command-line
      - uvicorn app.service.main:app --reload --host 0.0.0.0 --port 8000
      - http://localhost:81 (use port 81 to avoid conflicts with Docker container)
  - Debugging Python
    Create a .vscode directory in your workspace if it doesn't already exist.
    Create a launch.json file inside the .vscode directory (the one checked in should work)

- Run Aqua app in a Docker container (things that needed to be done)
  1. Create entrypoint.sh in root directory (for diagnostics)
      chmod +x entrypoint.sh

      The container might be exiting immediately because the CMD command is not keeping the container running. This can happen if either Nginx or Uvicorn fails to start, or if they start in the background and the main process exits.

      This was not the case - but it provides a way to get some diagnostic steps in the process. Added an nginx validation to the entrypoint shell
  2. nginx_aqua_docker.conf was not valid as a global nginx.conf file... which is how it was being setup in the dockerfile (thank you copilot). What needed to happen was that a global nginx.conf file needed to be created that includes the nginx_aqua_docker.conf from a conf.d directory. This is location typically used in containers for nginx app conf similar to the sites-enabled directory. But these configs are global rather than site specific. It's more straight forward and if it works for the container it's the way to go.
  3. libglibgl1-mesa-glx and libglib2.0-0 are dependencies for opencv (cv2 - used by the app). It was not included as part of the python virtual env because it was already installed for the OS. This should be installed as part of the dockerfile. (Could it be installed in the venv?)
- Install Docker Engine (https://docs.docker.com/engine/install/ubuntu/)
  1. for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
  2. add the docker respository to apt sources
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
  3. Install Docker
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  4. Test
    sudo docker run hello-world

- Install the Docker VSCode extension
  Future - the rootless operation material and set that up https://docs.docker.com/engine/security/rootless/
- Create Dockerfile in project root
- Create requirements.txt in service directory
- Create a docker specific nginx config file appinfra/nginx_aqua_docker.conf
- Determine if rootless docker would eliminate need to run docker commands as sudo
- Build the docker image
    cd /home/ubuntu/src/examples/aqua
    sudo docker build -t aqua-app .
- Run the docker image
    sudo docker run -d -p 80:80 aqua-app

- Debug app running in container
  - add debugpy to requirements.txt
  - expose port 5678 in Dockerfile
  - modify main.py to start the debugpy server
  - launch.json requires a config to attach to debugger
  - build and run
    cd /home/ubuntu/src/examples/aqua
    sudo docker build -t aqua-app .
    sudo docker run -d -p 80:80 -p 5678:5678 -e DEBUG_MODE=true aqua-app

  - to see built images
    sudo docker images
  - to see running containers
    sudo docker ps
  - to see running and stopped containers
    sudo docker ps -a
  - to see container logs
    sudo docker logs <container_id> | <name>
  - stop container
    sudo docker stop <container_id> | <name>


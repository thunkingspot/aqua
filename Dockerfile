# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /approot

# Copy the current directory contents into the container at /approot
COPY . /approot

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r app/service/requirements.txt

# Install additional dependencies
RUN apt-get update && apt-get install -y nginx libgl1-mesa-glx libglib2.0-0 libegl1-mesa-dev libgles2-mesa-dev mesa-common-dev
#RUN apt-get update && apt-get install -y nginx libgl1-mesa-glx libglib2.0-0 libosmesa6-dev

# Copy main Nginx configuration file
COPY appinfra/nginx.conf /etc/nginx/nginx.conf

# Copy custom Nginx configuration file
COPY appinfra/nginx_aqua_docker.conf /etc/nginx/conf.d/nginx_aqua_docker.conf

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Expose port 80 to the outside world
EXPOSE 80 5678

# Use the entrypoint script to start Nginx and Uvicorn
ENTRYPOINT ["/entrypoint.sh"]
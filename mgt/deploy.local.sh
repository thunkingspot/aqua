set -e

#recursively remove contents of /var/www/aqua_app and copy app/website into it
sudo rm -rf /var/www/aqua_app/*
sudo cp -r /home/ubuntu/src/aqua/app/website/* /var/www/aqua_app/

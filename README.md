# challenge-parking-lot-detection

This repository is organised as follows:
- challenge-parking-lot-detection.ipynb is the notebook that was used to train the Yolov7 network on the PK Lot public dataset;
- app.py is a python file where an API was developed in Flask that will allow this solution to be run in the future to detect occupied and empty spaces in a parking lot;
- report.pdf contains the documentation on the development of this project;
- Dockerfile.txt is a Dockerfile that will allow you to create an image of a Docker container;
- Docker-compose.yml is a YAML file containing the instructions so that the "docker compose up" command can create the Docker container and start our service;
- best.pt is the file that contains the weights saved after training Yolov7 in the PK Lot dataset;
- The models and utils directories contain files that have been extracted from the Yolov7 repository (https://github.com/WongKinYiu/yolov7), which are essential for this solution to work.


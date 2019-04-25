# Thai_dessert_detection_API

## download model
```bash
https://drive.google.com/file/d/1k7OWTbfMOF7hWG01IFP2_oHKt9QuR7Ps/view?usp=sharing
```
1. move model file to service folder 

## Build docker images and run container
you can build and run dockerfile using build-docker.sh 
```bash
./build-docker.sh
```
my shell script will install and setting nginx server for project

## Port and Rest api
1. 0.0.0.0:8080 or localhost:8080 ## initial page
2. 0.0.0.0:8080/predict or localhost:8080/predict ##this api for prediction object detection just POST image data to this api

# Let's Enjoy!!




#!/bin/bash
# build and push
URL='stanfordvl/surreal-cpu'

sudo docker build -t $URL:`cat VERSION` .
sudo docker tag $URL:`cat VERSION` $URL:latest
sudo docker push $URL:`cat VERSION`
sudo docker push $URL:latest

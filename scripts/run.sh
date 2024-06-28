#!/bin/bash
docker run -dit --gpus all --privileged -v ~/snntorch_network/:/root/snntorch_network/ \
	   -p 127.0.0.1:80:8080/tcp --network=host -e DISPLAY=$DISPLAY \
	   -v /tmp/.X11-unix/:/tmp/.X11-unix/ --name snntorch franzhd/snntorch bash \
#!/bin/bash

echo "building $image_version docker with image name franzhd/lava"

docker build --no-cache --tag  franzhd/snntorch .
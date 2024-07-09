#!/bin/bash

echo "building $image_version docker with image name franzhd/snntorch"

docker build --no-cache --tag  franzhd/snntorch .
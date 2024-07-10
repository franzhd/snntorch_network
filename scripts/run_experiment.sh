#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 --experiment_folder_path <path> --port <port>"
    exit 1
fi

docker run -dit --rm --gpus all --privileged --ipc=host \
	   -v ~/snntorch_network/:/root/snntorch_network/ \
	   --network=host -e DISPLAY=$DISPLAY \
	   -v /tmp/.X11-unix/:/tmp/.X11-unix/ --name snntorch franzhd/snntorch \
	   /bin/bash -c "nnictl create --config /root/snntorch_network/nni_experiments/$1/config/config_1.yml -p $2; sleep 2; ngrok http $2 >> /root/snntorch_network/scripts/ngrok_url.txt"

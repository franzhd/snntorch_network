#!/bin/bash

docker exec -it snntorch jupyter notebook --ip 0.0.0.0 \
            --no-browser --allow-root \
            --PasswordIdentityProvider.hashed_password='' \
            --IdentityProvider.token='snntorch' --ServerApp.allow_remote_access=True
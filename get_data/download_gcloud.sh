#!/bin/bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-439.0.0-linux-x86_64.tar.gz
tar zxvf google-cloud-sdk-439.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
exec -l $SHELL

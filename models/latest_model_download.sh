#!/bin/bash
#USAGE : bash latest_model_download.sh

#Create download directory
SAVEDIR="trained_model_latest"
mkdir -p $SAVEDIR

#Data
echo "START downloading."
echo;
aws s3 sync
echo;
echo "FINISH downloading."

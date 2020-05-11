#!/bin/bash 
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6



alias python=python3


pip install torch torchvision
pip install PIL

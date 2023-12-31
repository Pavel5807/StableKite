#!/bin/bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-pip gcc
sudo apt-get install -y python3-opencv
sudo apt-get install -y python3-numpy
sudo apt-get install git
sudo apt-get install wget
sudo apt-get install python3-setuptools
wget https://github.com/rockchip-linux/rknpu2/raw/master/runtime/RK356X/Linux/librknn_api/aarch64/librknnrt.so
sudo mv librknnrt.so /usr/lib/librknnrt.so
cd ~ && git clone https://github.com/rockchip-linux/rknn-toolkit2.git
cd rknn-toolkit2/rknn_toolkit_lite2/packages/ && pip3 install rknn_toolkit_lite2-1.5.2-cp310-cp310-linux_aarch64.whl

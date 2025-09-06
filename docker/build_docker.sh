#!/bin/bash
set -x

TAG=zheng/openvla-oft
PARENT=nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
USER_ID=`id -u`
GROUP_ID=`id -g`

docker build -f docker/Dockerfile \
  --build-arg PARENT_IMAGE=${PARENT} \
  --build-arg USER_ID=${USER_ID} \
  --build-arg GROUP_ID=${GROUP_ID} \
  -t ${TAG} .
#!/usr/bin/env bash

./build.sh

docker save sts24_algorithm_docker | gzip -c > sts24_algorithm_docker.tar.gz

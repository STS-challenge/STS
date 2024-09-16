#!/usr/bin/env bash

./build.sh

docker save STS2024_Algorithm_Docker | gzip -c > STS2024_Algorithm_Docker.tar.gz

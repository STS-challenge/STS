#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t sts24_algorithm_docker "$SCRIPTPATH"

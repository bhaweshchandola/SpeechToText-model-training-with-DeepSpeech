#!/bin/bash

set -xe

cuda=$1

source $(dirname "$0")/tc-tests-utils.sh

if [ "${cuda}" = "--cuda" ]; then
    PROJECT_NAME="DeepSpeech-GPU"
else
    PROJECT_NAME="DeepSpeech"
fi

download_data

install_nuget "${PROJECT_NAME}"

run_netframework_inference_tests

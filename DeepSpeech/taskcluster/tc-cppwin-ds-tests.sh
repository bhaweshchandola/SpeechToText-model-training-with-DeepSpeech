#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

download_material "${TASKCLUSTER_TMP_DIR}/ds"

export PATH=${TASKCLUSTER_TMP_DIR}/ds/:$PATH

check_tensorflow_version

run_basic_inference_tests

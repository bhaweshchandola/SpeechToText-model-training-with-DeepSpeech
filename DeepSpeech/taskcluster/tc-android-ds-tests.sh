#!/bin/bash

set -xe

arm_flavor=$1
api_level=$2

source $(dirname "$0")/tc-tests-utils.sh

model_source=${DEEPSPEECH_TEST_MODEL//.pb/.tflite}
model_name=$(basename "${model_source}")

if [ "${arm_flavor}" = "armeabi-v7a" ]; then
    export DEEPSPEECH_ARTIFACTS_ROOT=${DEEPSPEECH_ARTIFACTS_ROOT_ARMV7}
fi

if [ "${arm_flavor}" = "arm64-v8a" ]; then
    export DEEPSPEECH_ARTIFACTS_ROOT=${DEEPSPEECH_ARTIFACTS_ROOT_ARM64}
fi

download_material "${TASKCLUSTER_TMP_DIR}/ds"

android_setup_emulator "${arm_flavor}" "${api_level}"

android_setup_ndk_data

check_tensorflow_version

run_tflite_basic_inference_tests

android_stop_emulator

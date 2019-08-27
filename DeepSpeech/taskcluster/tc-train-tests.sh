#!/bin/bash

set -xe

source $(dirname "$0")/tc-tests-utils.sh

pyver_full=$1
ds=$2

if [ -z "${pyver_full}" ]; then
    echo "No python version given, aborting."
    exit 1
fi;

pyver=$(echo "${pyver_full}" | cut -d':' -f1)

# 2.7.x => 27
pyver_pkg=$(echo "${pyver}" | cut -d'.' -f1,2 | tr -d '.')

py_unicode_type=$(echo "${pyver_full}" | cut -d':' -f2)
if [ "${py_unicode_type}" = "m" ]; then
  pyconf="ucs2"
elif [ "${py_unicode_type}" = "mu" ]; then
  pyconf="ucs4"
fi;

unset PYTHON_BIN_PATH
unset PYTHONPATH
export PYENV_ROOT="${HOME}/ds-train/.pyenv"
export PATH="${PYENV_ROOT}/bin:${HOME}/bin:$PATH"

mkdir -p ${PYENV_ROOT} || true
mkdir -p ${TASKCLUSTER_ARTIFACTS} || true
mkdir -p /tmp/train || true
mkdir -p /tmp/train_tflite || true

install_pyenv "${PYENV_ROOT}"
install_pyenv_virtualenv "$(pyenv root)/plugins/pyenv-virtualenv"

PYENV_NAME=deepspeech-train
PYTHON_CONFIGURE_OPTS="--enable-unicode=${pyconf}" pyenv install ${pyver}
pyenv virtualenv ${pyver} ${PYENV_NAME}
source ${PYENV_ROOT}/versions/${pyver}/envs/${PYENV_NAME}/bin/activate

pip install --upgrade -r ${HOME}/DeepSpeech/ds/requirements.txt | cat

pushd ${HOME}/DeepSpeech/ds/
    verify_ctcdecoder_url
popd

platform=$(python -c 'import sys; import platform; plat = platform.system().lower(); arch = platform.machine().lower(); plat = "manylinux1" if plat == "linux" and arch == "x86_64" else plat; plat = "macosx_10_10" if plat == "darwin" else plat; sys.stdout.write("%s_%s" % (plat, platform.machine()));')
whl_ds_version="$(python -c 'from pkg_resources import parse_version; print(parse_version("'${DS_VERSION}'"))')"
decoder_pkg="ds_ctcdecoder-${whl_ds_version}-cp${pyver_pkg}-cp${pyver_pkg}${py_unicode_type}-${platform}.whl"

decoder_pkg_url=${DECODER_ARTIFACTS_ROOT}/${decoder_pkg}

LD_LIBRARY_PATH=${PY37_LDPATH}:$LD_LIBRARY_PATH pip install --verbose --only-binary :all: ${PY37_SOURCE_PACKAGE} ${decoder_pkg_url} | cat

pushd ${HOME}/DeepSpeech/ds/
    # Run twice to test preprocessed features
    time ./bin/run-tc-ldc93s1_new.sh 199
    time ./bin/run-tc-ldc93s1_new.sh 200
    time ./bin/run-tc-ldc93s1_tflite.sh
popd

cp /tmp/train/output_graph.pb ${TASKCLUSTER_ARTIFACTS}
cp /tmp/train_tflite/output_graph.tflite ${TASKCLUSTER_ARTIFACTS}

if [ ! -z "${CONVERT_GRAPHDEF_MEMMAPPED}" ]; then
  convert_graphdef=$(basename "${CONVERT_GRAPHDEF_MEMMAPPED}")
  wget -P "/tmp/" "${CONVERT_GRAPHDEF_MEMMAPPED}" && chmod +x "/tmp/${convert_graphdef}"

  /tmp/${convert_graphdef} --in_graph=/tmp/train/output_graph.pb --out_graph=/tmp/train/output_graph.pbmm
  cp /tmp/train/output_graph.pbmm ${TASKCLUSTER_ARTIFACTS}
fi;

pushd ${HOME}/DeepSpeech/ds/
    time ./bin/run-tc-ldc93s1_checkpoint.sh 200
popd

deactivate
pyenv uninstall --force ${PYENV_NAME}

#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python3.5 -u DeepSpeech.py \
  --train_files /usr/workspace/pykaldi/deepspeech/vietnam/deepspeech_training/train/train.csv \
#   --dev_files /usr/workspace/pykaldi/deepspeech/vietnam/deepspeech_training/dev/dev.csv \
  --test_files /usr/workspace/pykaldi/deepspeech/vietnam/deepspeech_training/test/test.csv \
  --train_batch_size 80 \
  --dev_batch_size 80 \
  --test_batch_size 40 \
  --n_hidden 375 \
  --epoch 33 \
  --validation_step 1 \
  --early_stop True \
  --earlystop_nsteps 6 \
  --estop_mean_thresh 0.1 \
  --estop_std_thresh 0.1 \
  --dropout_rate 0.22 \
  --learning_rate 0.00095 \
  --report_count 100 \
  --use_seq_length False \
  --export_dir /usr/workspace/pykaldi/deepspeech/vietnam/deepspeech_training/results/model_export/ \
  --checkpoint_dir /usr/workspace/pykaldi/deepspeech/vietnam/deepspeech_training/results/checkout/ \
  --decoder_library_path /home/nvidia/tensorflow/bazel-bin/native_client/libctc_decoder_with_kenlm.so \
  --alphabet_config_path /usr/workspace/pykaldi/deepspeech/vietnam/deepspeech_training/alphabet.txt \
  --lm_binary_path /usr/workspace/pykaldi/deepspeech/vietnam/deepspeech_training/lm.binary \
  --lm_trie_path /usr/workspace/pykaldi/deepspeech/vietnam/deepspeech_training/trie \
  "$@"
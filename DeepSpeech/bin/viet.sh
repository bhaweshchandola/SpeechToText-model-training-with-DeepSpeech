#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

python3.5 -u DeepSpeech.py \
  --train_batch_size 80 \
  --test_batch_size 40 \
  --n_hidden 375 \
  --epoch 3 \
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
  --alphabet_config_path /usr/workspace/pykaldi/deepspeech/vietnam/deepspeech_training/alphabet.txt \
  --lm_binary_path /usr/workspace/pykaldi/deepspeech/vietnam/deepspeech_training/lm.bin \
  --lm_trie_path /usr/workspace/pykaldi/deepspeech/vietnam/deepspeech_training/trie \
  "$@"

from deepspeech import Model
import scipy.io.wavfile
import scipy.io.wavfile as wav
import sys
try:
    from shhlex import quote
except ImportError:
    from pipes import quote
import subprocess
import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave

BEAM_WIDTH = 500
LM_WEIGHT = 1.50
VALID_WORD_COUNT_WEIGHT = 2.25
N_FEATURES = 26
N_CONTEXT = 9
# MODEL_FILE = 'output_graph.pbmm'
# ALPHABET_FILE = 'alphabet.txt'
# LANGUAGE_MODEL =  'lm.binary'
# TRIE_FILE =  'trie'

def convert_samplerate(audio_path):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate 16000 --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path))
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use 16kHz files or install it: {}'.format(e.strerror))

    return 16000, np.frombuffer(output, np.int16)

# alphabet = "/usr/workspace/model_sdcs_5_10_2018/models/alphabet.txt"
alphabet = "alphabet.txt"
# lm = "/usr/workspace/model_sdcs_5_10_2018/models/lm.binary"
lm = "lm.bin"
# trie = "/usr/workspace/model_sdcs_5_10_2018/models/trie"
trie = "trie"
# ds = Model("/usr/workspace/model_sdcs_5_10_2018/models/output_graph.pb", 26, 9, alphabet, 500)
ds = Model("output_graph.pb", 26, 9, alphabet, 500)

# ds = Model(MODEL_FILE, N_FEATURES, N_CONTEXT, ALPHABET_FILE, BEAM_WIDTH)

ds.enableDecoderWithLM(alphabet, lm, trie, LM_WEIGHT, 
VALID_WORD_COUNT_WEIGHT)

def process(path):
    # fs, audio = scipy.io.wavfile.read(path)
    fs, audio = convert_samplerate(path)
    processed_data = ds.stt(audio, fs)
    return processed_data   

print(process('temp_123.wav'))
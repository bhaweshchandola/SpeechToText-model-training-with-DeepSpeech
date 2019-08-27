#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import argparse
import os
import sys


sys.path.insert(1, os.path.join(sys.path[0], '..'))

import csv
import re
import sox
import zipfile
import subprocess
import progressbar
import unicodedata

from threading import RLock
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from util.downloader import SIMPLE_BAR

from os import path
from glob import glob

from util.downloader import maybe_download
from util.text import Alphabet, validate_label
from util.feeding import secs_to_hours

FIELDNAMES = ['wav_filename', 'wav_filesize', 'transcript']
SAMPLE_RATE = 16000
MAX_SECS = 10

ARCHIVE_DIR_NAME = 'lingua_libre'
ARCHIVE_NAME = 'Q{qId}-{iso639_3}-{language_English_name}.zip'
ARCHIVE_URL = 'https://lingualibre.fr/datasets/' + ARCHIVE_NAME


def _download_and_preprocess_data(target_dir):
    # Making path absolute
    target_dir = path.abspath(target_dir)
    # Conditionally download data
    archive_path = maybe_download(ARCHIVE_NAME, target_dir, ARCHIVE_URL)
    # Conditionally extract data
    _maybe_extract(target_dir, ARCHIVE_DIR_NAME, archive_path)
    # Produce CSV files and convert ogg data to wav
    _maybe_convert_sets(target_dir, ARCHIVE_DIR_NAME)

def _maybe_extract(target_dir, extracted_data, archive_path):
    # If target_dir/extracted_data does not exist, extract archive in target_dir
    extracted_path = path.join(target_dir, extracted_data)
    if not path.exists(extracted_path):
        print('No directory "%s" - extracting archive...' % extracted_path)
        if not os.path.isdir(extracted_path):
            os.mkdir(extracted_path)
        with zipfile.ZipFile(archive_path) as zip_f:
            zip_f.extractall(extracted_path)
    else:
        print('Found directory "%s" - not extracting it from archive.' % archive_path)

def _maybe_convert_sets(target_dir, extracted_data):
    extracted_dir = path.join(target_dir, extracted_data)
    # override existing CSV with normalized one
    target_csv_template = os.path.join(target_dir, ARCHIVE_DIR_NAME + '_' + ARCHIVE_NAME.replace('.zip', '_{}.csv'))
    if os.path.isfile(target_csv_template):
        return

    ogg_root_dir = os.path.join(extracted_dir, ARCHIVE_NAME.replace('.zip', ''))

    # Get audiofile path and transcript for each sentence in tsv
    samples = []
    glob_dir = os.path.join(ogg_root_dir, '**/*.ogg')
    for record in glob(glob_dir, recursive=True):
        record_file = record.replace(ogg_root_dir + os.path.sep, '')
        if record_filter(record_file):
            samples.append((record_file, os.path.splitext(os.path.basename(record_file))[0]))

    # Keep track of how many samples are good vs. problematic
    counter = {'all': 0, 'failed': 0, 'invalid_label': 0, 'too_short': 0, 'too_long': 0, 'total_time': 0}
    lock = RLock()
    num_samples = len(samples)
    rows = []

    def one_sample(sample):
        """ Take a audio file, and optionally convert it to 16kHz WAV """
        ogg_filename = path.join(ogg_root_dir, sample[0])
        # Storing wav files next to the ogg ones - just with a different suffix
        wav_filename = path.splitext(ogg_filename)[0] + ".wav"
        _maybe_convert_wav(ogg_filename, wav_filename)
        file_size = -1
        frames = 0
        if path.exists(wav_filename):
            file_size = path.getsize(wav_filename)
            frames = int(subprocess.check_output(['soxi', '-s', wav_filename], stderr=subprocess.STDOUT))
        label = label_filter(sample[1])
        with lock:
            if file_size == -1:
                # Excluding samples that failed upon conversion
                counter['failed'] += 1
            elif label is None:
                # Excluding samples that failed on label validation
                counter['invalid_label'] += 1
            elif int(frames/SAMPLE_RATE*1000/10/2) < len(str(label)):
                # Excluding samples that are too short to fit the transcript
                counter['too_short'] += 1
            elif frames/SAMPLE_RATE > MAX_SECS:
                # Excluding very long samples to keep a reasonable batch-size
                counter['too_long'] += 1
            else:
                # This one is good - keep it for the target CSV
                rows.append((wav_filename, file_size, label))
            counter['all'] += 1
            counter['total_time'] += frames

    print("Importing ogg files...")
    pool = Pool(cpu_count())
    bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
    for i, _ in enumerate(pool.imap_unordered(one_sample, samples), start=1):
        bar.update(i)
    bar.update(num_samples)
    pool.close()
    pool.join()

    with open(target_csv_template.format('train'), 'w') as train_csv_file:  # 80%
        with open(target_csv_template.format('dev'), 'w') as dev_csv_file:  # 10%
            with open(target_csv_template.format('test'), 'w') as test_csv_file:  # 10%
                train_writer = csv.DictWriter(train_csv_file, fieldnames=FIELDNAMES)
                train_writer.writeheader()
                dev_writer = csv.DictWriter(dev_csv_file, fieldnames=FIELDNAMES)
                dev_writer.writeheader()
                test_writer = csv.DictWriter(test_csv_file, fieldnames=FIELDNAMES)
                test_writer.writeheader()

                for i, item in enumerate(rows):
                    transcript = validate_label(item[2])
                    if not transcript:
                        continue
                    wav_filename = os.path.join(ogg_root_dir, item[0].replace('.ogg', '.wav'))
                    i_mod = i % 10
                    if i_mod == 0:
                        writer = test_writer
                    elif i_mod == 1:
                        writer = dev_writer
                    else:
                        writer = train_writer
                    writer.writerow(dict(
                        wav_filename=wav_filename,
                        wav_filesize=os.path.getsize(wav_filename),
                        transcript=transcript,
                    ))

    print('Imported %d samples.' % (counter['all'] - counter['failed'] - counter['too_short'] - counter['too_long']))
    if counter['failed'] > 0:
        print('Skipped %d samples that failed upon conversion.' % counter['failed'])
    if counter['invalid_label'] > 0:
        print('Skipped %d samples that failed on transcript validation.' % counter['invalid_label'])
    if counter['too_short'] > 0:
        print('Skipped %d samples that were too short to match the transcript.' % counter['too_short'])
    if counter['too_long'] > 0:
        print('Skipped %d samples that were longer than %d seconds.' % (counter['too_long'], MAX_SECS))
    print('Final amount of imported audio: %s.' % secs_to_hours(counter['total_time'] / SAMPLE_RATE))

def _maybe_convert_wav(ogg_filename, wav_filename):
    if not path.exists(wav_filename):
        transformer = sox.Transformer()
        transformer.convert(samplerate=SAMPLE_RATE)
        try:
            transformer.build(ogg_filename, wav_filename)
        except sox.core.SoxError as ex:
            print('SoX processing error', ex, ogg_filename, wav_filename)

def handle_args():
    parser = argparse.ArgumentParser(description='Importer for LinguaLibre dataset. Check https://lingualibre.fr/wiki/Help:Download_from_LinguaLibre for details.')
    parser.add_argument(dest='target_dir')
    parser.add_argument('--qId', type=int, required=True, help='LinguaLibre language qId')
    parser.add_argument('--iso639-3', type=str, required=True, help='ISO639-3 language code')
    parser.add_argument('--english-name', type=str, required=True, help='Enligh name of the language')
    parser.add_argument('--filter_alphabet', help='Exclude samples with characters not in provided alphabet')
    parser.add_argument('--normalize', action='store_true', help='Converts diacritic characters to their base ones')
    parser.add_argument('--bogus-records', type=argparse.FileType('r'), required=False, help='Text file listing well-known bogus record to skip from importing, from https://lingualibre.fr/wiki/LinguaLibre:Misleading_items')
    return parser.parse_args()

if __name__ == "__main__":
    CLI_ARGS = handle_args()
    ALPHABET = Alphabet(CLI_ARGS.filter_alphabet) if CLI_ARGS.filter_alphabet else None

    bogus_regexes = []
    if CLI_ARGS.bogus_records:
        for line in CLI_ARGS.bogus_records:
            bogus_regexes.append(re.compile(line.strip()))

    def record_filter(path):
        if any(regex.match(path) for regex in bogus_regexes):
            print('Reject', path)
            return False
        return True

    def label_filter(label):
        if CLI_ARGS.normalize:
            label = unicodedata.normalize("NFKD", label.strip()) \
                .encode("ascii", "ignore") \
                .decode("ascii", "ignore")
        label = validate_label(label)
        if ALPHABET and label:
            try:
                [ALPHABET.label_from_string(c) for c in label]
            except KeyError:
                label = None
        return label

    ARCHIVE_NAME = ARCHIVE_NAME.format(qId=CLI_ARGS.qId, iso639_3=CLI_ARGS.iso639_3, language_English_name=CLI_ARGS.english_name)
    ARCHIVE_URL  = ARCHIVE_URL.format(qId=CLI_ARGS.qId, iso639_3=CLI_ARGS.iso639_3, language_English_name=CLI_ARGS.english_name)
    _download_and_preprocess_data(target_dir=CLI_ARGS.target_dir)

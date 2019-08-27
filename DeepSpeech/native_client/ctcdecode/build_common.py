#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import glob
import os
import shlex
import subprocess
import sys

from multiprocessing.dummy import Pool

ARGS = ['-O3', '-DNDEBUG', '-DKENLM_MAX_ORDER=6', '-std=c++11',
        '-Wno-unused-local-typedefs', '-Wno-sign-compare']

INCLUDES = [
    '..',
    '../kenlm',
    'third_party/openfst-1.6.7/src/include',
    'third_party/ThreadPool'
]

COMMON_FILES = (glob.glob('../kenlm/util/*.cc')
                + glob.glob('../kenlm/lm/*.cc')
                + glob.glob('../kenlm/util/double-conversion/*.cc'))

COMMON_FILES += glob.glob('third_party/openfst-1.6.7/src/lib/*.cc')

COMMON_FILES = [
    fn for fn in COMMON_FILES
    if not (fn.endswith('main.cc') or fn.endswith('test.cc') or fn.endswith(
        'unittest.cc'))
]

COMMON_FILES += glob.glob('*.cpp')

def build_common(out_name='common.a', build_dir='temp_build/temp_build', num_parallel=1):
    compiler = os.environ.get('CXX', 'g++')
    ar = os.environ.get('AR', 'ar')
    libtool = os.environ.get('LIBTOOL', 'libtool')
    cflags = os.environ.get('CFLAGS', '') + os.environ.get('CXXFLAGS', '')

    for file in COMMON_FILES:
        outfile = os.path.join(build_dir, os.path.splitext(file)[0] + '.o')
        outdir = os.path.dirname(outfile)
        if not os.path.exists(outdir):
            print('mkdir', outdir)
            os.makedirs(outdir)

    def build_one(file):
        outfile = os.path.join(build_dir, os.path.splitext(file)[0] + '.o')
        if os.path.exists(outfile):
            return

        cmd = '{cc} -fPIC -c {cflags} {args} {includes} {infile} -o {outfile}'.format(
            cc=compiler,
            cflags=cflags,
            args=' '.join(ARGS),
            includes=' '.join('-I' + i for i in INCLUDES),
            infile=file,
            outfile=outfile,
        )
        print(cmd)
        subprocess.check_call(shlex.split(cmd))
        return outfile

    pool = Pool(num_parallel)
    obj_files = list(pool.imap_unordered(build_one, COMMON_FILES))

    if sys.platform.startswith('darwin'):
        cmd = '{libtool} -static -o {outfile} {infiles}'.format(
            libtool=libtool,
            outfile=out_name,
            infiles=' '.join(obj_files),
        )
        print(cmd)
        subprocess.check_call(shlex.split(cmd))
    else:
        cmd = '{ar} rcs {outfile} {infiles}'.format(
            ar=ar,
            outfile=out_name,
            infiles=' '.join(obj_files)
        )
        print(cmd)
        subprocess.check_call(shlex.split(cmd))

if __name__ == '__main__':
    build_common()

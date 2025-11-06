# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# --------------------------------------------------------

import os
import numpy as np
import logging 

VERBOSE = True

def mkdir(out_dir, info=True):
    if VERBOSE and info and not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

def print_info(*objects, log = True, sep = ' '):
    if not VERBOSE: return
    info = sep.join([str(obj) for obj in objects])
    print(info)
    if log:
        logging.info(info)

def exists(val):
    return val is not None

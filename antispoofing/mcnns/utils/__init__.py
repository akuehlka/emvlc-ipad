# -*- coding: utf-8 -*-

# -- common functions and constants
from antispoofing.mcnns.utils.constants import N_JOBS
from antispoofing.mcnns.utils.constants import CONST
from antispoofing.mcnns.utils.constants import PROJECT_PATH
from antispoofing.mcnns.utils.constants import UTILS_PATH
from antispoofing.mcnns.utils.constants import SEED


from antispoofing.mcnns.utils.misc import modification_date
from antispoofing.mcnns.utils.misc import get_time
from antispoofing.mcnns.utils.misc import total_time_elapsed
from antispoofing.mcnns.utils.misc import RunInParallel
from antispoofing.mcnns.utils.misc import progressbar
from antispoofing.mcnns.utils.misc import save_object
from antispoofing.mcnns.utils.misc import load_object
from antispoofing.mcnns.utils.misc import load_images
from antispoofing.mcnns.utils.misc import mosaic
from antispoofing.mcnns.utils.misc import read_csv_file
from antispoofing.mcnns.utils.misc import get_interesting_samples
from antispoofing.mcnns.utils.misc import create_mosaic

# -- common imports
from sys import platform
import numpy as np
np.random.rand(SEED)

import pdb

import matplotlib
if platform == 'linux':
    matplotlib.use('Agg')
if platform == 'darwin':
    matplotlib.use('qt5agg')

# -*- coding: utf-8 -*-

import os
from multiprocessing import cpu_count

SEED = 0

N_JOBS = (cpu_count() - 1) if ((cpu_count()) > 1) else 1
N_JOBS = min(N_JOBS, 5)

PROJECT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
UTILS_PATH = os.path.dirname(__file__)


class CONST:

    def __init__(self):
        pass

    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import random
from distutils import dir_util
import os
import shutil
import subprocess
import netCDF4 as nc
import glob
# from .test_utils import

from aquacrop import run


@pytest.fixture
def context(tmpdir, request):
    """Run AquaCrop model"""
    # this part of the function searches for a directory with
    # the same name of test module and, if available, moving all
    # contents to a temporary directory so tests can use them
    # freely (https://stackoverflow.com/a/29631801)
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)
    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    # test_yr = str(1979)
    # test_yr = str(2000)
    yrs = os.walk(str(tmpdir)).__next__()[1]
    test_yr = yrs[0]
    cwd = os.getcwd()
    aquacrop_dir = os.path.abspath("../..")
    for yr in yrs:
        if yr == test_yr:
            os.chdir(os.path.join(str(tmpdir), str(yr)))
            # runner = os.path.join(aquacrop_dir, "aquacrop", "run.py")
            config = glob.glob("Config/*" + "_config.ini")[0]
            run.main([config])
            # print(runner)
            # print(config)
            # try:
            #     subprocess.check_call([runner, config])
            # except subprocess.CalledProcessError:
            #     raise
            os.chdir(cwd)
        else:
            shutil.rmtree(os.path.join(str(tmpdir), str(yr)))

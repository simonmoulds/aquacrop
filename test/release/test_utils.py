#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import netCDF4 as nc
import pandas
import re

def read_aqpy_yield(fn):
    d = nc.Dataset(fn)
    yld = d.variables['crop_yield'][0,0,:,0,0]
    yld = np.max(yld)
    return yld

def read_aos_yield(fn):
    d = pandas.read_csv(fn, delimiter='\s+|\t', engine='python')
    yld = d['Yield'].values
    yld = np.max(yld)
    return yld

def read_yield(datadir):
    datadir = str(datadir)
    yrs = os.walk(datadir).__next__()[1]
    aqpy_yld = np.zeros(len(yrs))
    aos_yld = np.zeros(len(yrs))
    for ix, yr in enumerate(yrs):
        fn = os.path.join(datadir, str(yr), "Output", "netcdf", "Y_cropland_dailyTot_output.nc")
        try:
            aqpy_yld[ix] = read_aqpy_yield(fn)
        except:
            raise
        exp_output_dir = os.path.join(datadir, str(yr), "Expected_Output")
        fn = [os.path.join(exp_output_dir, f) for f in os.listdir(exp_output_dir) if re.match(".*_FinalOutput.txt", f)]
        if (len(fn) > 0):
            fn = fn[0]
            try:
                aos_yld[ix] = read_aos_yield(fn)
            except:
                raise
        else:
            msg = 'AquaCropOS output file not found'
            raise FileNotFoundError(msg)

    return aqpy_yld, aos_yld

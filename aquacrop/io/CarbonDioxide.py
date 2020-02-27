#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from hm.input import HmInputData

import logging
logger = logging.getLogger(__name__)

refconc = 369.41

class CarbonDioxide(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = \
            self.model.config.CARBON_DIOXIDE['carbonDioxideNC']
        self.nc_varname = \
            self.model.config.CARBON_DIOXIDE['carbonDioxideVarName']
        self.model_varname = 'conc'

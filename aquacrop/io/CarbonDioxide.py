#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hm.input import HmInputData

import logging
logger = logging.getLogger(__name__)

refconc = 369.41


class CarbonDioxide(HmInputData):
    def __init__(self, model):
        self.model = model
        self.filename = \
            model.config.CARBON_DIOXIDE['filename']
        self.nc_varname = \
            model.config.CARBON_DIOXIDE['varname']
        self.is_1d = model.config.CARBON_DIOXIDE['is_1d']
        self.xy_dimname = model.config.CARBON_DIOXIDE['xy_dimname']
        self.model_varname = 'conc'

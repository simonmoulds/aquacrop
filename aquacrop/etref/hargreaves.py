#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from hm.model import Model
from hm.reporting import Reporting

from .extraterrestrialradiation import ExtraterrestrialRadiation
from .inputdata import Temperature
from . import variable_list

import logging
logger = logging.getLogger(__name__)

class Hargreaves(Model):

    def initial(self):
        self.temperature_module = Temperature(self)
        self.extraterrestrial_radiation_module = ExtraterrestrialRadiation(self)
        self.temperature_module.initial()
        self.extraterrestrial_radiation_module.initial()
        self.ETref = np.zeros((self.domain.nxy))        
        self.reporting_module = Reporting(self, variable_list, config_section='ETREF')
        self.reporting_module.initial()
        
    def hargreaves(self):
        self.ETref = (
            0.0023
            * (self.extraterrestrial_radiation * 0.408)  # MJ m-2 d-1 -> mm d-1
            * ((np.maximum(0, (self.tmean.values[0, self.domain.mask.values] - 273.0))) + 17.8)
            * np.sqrt(np.maximum(0, (self.tmax.values[0, self.domain.mask.values] - self.tmin.values[0, self.domain.mask.values])))
        )
    
    def dynamic(self):
        self.temperature_module.dynamic()
        self.extraterrestrial_radiation_module.dynamic()
        self.hargreaves()
        self.reporting_module.dynamic()

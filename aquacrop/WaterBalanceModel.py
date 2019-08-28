#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from hm import file_handling
from hm.Model import Model
from .Weather import Weather
from .Groundwater import Groundwater
from .LandSurface import LandSurface

import logging
logger = logging.getLogger(__name__)

class WaterBalanceModel(Model):    
    def __init__(self, configuration, modelTime, initialState = None):
        super(WaterBalanceModel, self).__init__(
            configuration,
            modelTime,
            initialState)

        self.weather_module = Weather(self)
        self.groundwater_module = Groundwater(self)
        # self.canal_module = CanalSupply(self)
        self.lc_module = LandSurface(self)
        
    def initial(self):
        self.weather_module.initial()
        self.groundwater_module.initial()
        # self.canal_module.initial()
        self.lc_module.initial()
        
    def dynamic(self):
        self.weather_module.dynamic()
        self.groundwater_module.dynamic()
        # self.canal_module.dynamic()
        self.lc_module.dynamic()

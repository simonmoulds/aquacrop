#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from hm.model import Model
from .io.Weather import Weather
from .io.Groundwater import Groundwater
from .LandSurface import LandSurface

import logging
logger = logging.getLogger(__name__)

class AquaCrop(Model):    
    def __init__(self, config, time, domain, init=None):
        super(AquaCrop, self).__init__(
            config,
            time,
            domain,
            is_1d=True,
            init=init
        )
        self.weather_module = Weather(self)
        self.groundwater_module = Groundwater(self)
        self.crop_module = LandSurface(self)
        
    def initial(self):
        self.weather_module.initial()
        self.groundwater_module.initial()
        self.crop_module.initial()
        
    def dynamic(self):
        self.weather_module.dynamic()
        self.groundwater_module.dynamic()
        self.crop_module.dynamic()

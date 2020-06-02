#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

class FarmParameters(object):

    def __init__(self, model):
        self.model = model
        self.config = self.model.config.FARM_PARAMETERS
        self.model.nFarm = len(self.model.domain._coords['farm'])
        self.model.AnnualChangeInFarmArea = self.config['AnnualChangeInFarmArea']

    def initial(self):
        pass

    # def set_farm_area(self):
    #     self.model.FarmArea = np.ones(
    #         (self.model.nFarm,
    #          self.model.nCrop,
    #          self.model.domain.nxy)
    #     )
        
    def dynamic(self):
        pass
        # self.set_farm_area()

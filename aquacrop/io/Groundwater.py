#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd

class Groundwater(object):

    def __init__(self, model):
        self.model = model
        self.WaterTable = self.model.config.WATER_TABLE['water_table']
        self.DynamicWaterTable = self.model.config.WATER_TABLE['dynamic']
        self.zGW = np.zeros(self.model.domain.nxy)
        self.coupled = self.model.config.WATER_TABLE['coupled']
        self.coupling_directory = self.model.config.WATER_TABLE['directory']
        self.time_lag = self.model.config.WATER_TABLE['time_lag']
        self.max_wait_time = self.model.config.WATER_TABLE['max_wait_time']
        self.wait_interval = self.model.config.WATER_TABLE['wait_interval']
        
    def initial(self):
        if self.WaterTable:            
            if self.coupled & (self.time_lag > 0):
                self.zGW = open_hmdataarray(
                    self.model.config.INITIAL_CONDITIONS['initialGroundwaterLevelNC'],
                    self.model.config.INITIAL_CONDITIONS['initialGroundwaterLevelVarName'],
                    self.model.domain,
                    self.model.config.INITIAL_CONDITIONS['is_1d'],
                    self.model.config.INITIAL_CONDITIONS['xy_dimname']
                )                

    def read(self):        
        if self.WaterTable:
            filename = self.model.config.WATER_TABLE['filename'].format(day=day, month=month, year=year)
            # if self.DynamicWaterTable:
            if self.coupled:
                # do not ask the model to read a file from a timestep prior to
                # the current simulation. 
                if not (self.model.time.isFirstTimestep() & (self.time_lag > 0)):                        
                    tm = self.model.time.curr_time - pd.Timedelta(self.time_lag)

                    # Check whether the file is present in the filesystem; if
                    # it doesn't, enter a while loop which periodically checks
                    # whether the file exists. We specify a maximum wait time
                    # in order to prevent the model hanging if the file never
                    # materialises.                        
                    exists = os.path.exists(filename)
                    total_wait_time = 0
                    while exists is False and total_wait_time <= self.max_wait_time:
                        time.sleep(self.wait_interval)
                        exists = os.path.exists(filename)
                        total_wait_time += self.wait_interval
                    if not exists:
                        msg = "groundwater file doesn't exist and maximum wait time exceeded"
                        raise OSError(
                            'groundwater file ' + filename + ' doesn\'t exist and maximum wait time exceeded'
                        )
                    
                    # if the simulation advances beyond the above control structure
                    # then filename must exist, so read the data.
                    self.zGW = hm.open_hmdataarray(
                        self.model.config.WATER_TABLE['filename'],
                        self.model.config.WATER_TABLE['varname'],
                        self.domain,
                        self.model.config.WATER_TABLE['is_1d'],
                        self.model.config.WATER_TABLE['xy_varname']
                    )
            else:
                self.zGW = hm.open_hmdataarray(
                    self.model.config.WATER_TABLE['filename'],
                    self.model.config.WATER_TABLE['varname'],
                    self.domain,
                    self.model.config.WATER_TABLE['is_1d'],
                    self.model.config.WATER_TABLE['xy_varname']
                )
                
        else:
            # if groundwater is not supplied, set depth to a large number
            # so that it doesn't influence root zone hydrological processes
            self.zGW += 9999.
            
    def dynamic(self):
        self.read()

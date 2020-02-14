#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import datetime as datetime

from hm import file_handling
from hm.Messages import ModelError, ModelFileError, ModelWarning

class Groundwater(object):

    def __init__(self, model):
        self.model = model
        # self._configuration = Groundwater_variable._configuration
        # self._modelTime = Groundwater_variable._modelTime
        # self.cloneMapFileName = Groundwater_variable.cloneMapFileName
        # self.landmask = Groundwater_variable.landmask
        
    def initial(self):
        self.WaterTable = bool(int(self.model.config.WATER_TABLE['WaterTable']))
        self.VariableWaterTable = bool(int(self.model.config.WATER_TABLE['VariableWaterTable']))
        self.DailyGroundwaterNC = bool(int(self.model.config.WATER_TABLE['DailyGroundwaterNC']))
        # TODO:
        zGW = np.ones_like(self.landmask) * 999.
        self.zGW = zGW[self.landmask]
        
        if self.WaterTable:            
            self.gwTimeLag = int(self.config.model.WATER_TABLE['timeLag'])
            if self.DailyGroundwaterNC & (self.gwTimeLag > 0):
                # if the program is configured to read daily groundwater files and
                # the time lag is positive, we also need to read an initial value
                # file
                self.zGW = open_hmdataarray(
                    self.model.config.INITIAL_CONDITIONS['initialGroundwaterLevelNC'],
                    self.model.config.WATER_TABLE['groundwaterVarName'],
                    self.model.domain
                )                
                # zGW = file_handling.netcdf_to_arrayWithoutTime(
                #     self._configuration.INITIAL_CONDITIONS['initialGroundwaterLevelNC'],
                #     self._configuration.WATER_TABLE['groundwaterVarName'],
                #     cloneMapFileName = self.cloneMapFileName
                # )
                # self.zGW = zGW[self.landmask]

    def read(self):
        
        if self.WaterTable:
            
            # Fill named placeholders (NB we have already checked that
            # the specified filename contains these placeholders)
            
            # gwFileNC = self._configuration.WATER_TABLE['groundwaterNC'].format(day=day, month=month, year=year)
            
            if self.VariableWaterTable:

                # DailyGroundwaterNC is a logical indicating whether a separate
                # netCDF is used for each time step - use this for coupling
                if self.DailyGroundwaterNC:

                    # introduce this test so that we do not ask the model to read
                    # a file from a timestep prior to the current simulation. 
                    if not (self.model.time.isFirstTimestep() & (self.gwTimeLag > 0)):                        
                        tm = self.model.time.curr_time - datetime.timedelta(self.gwTimeLag)
                        day, month, year = tm.day, tm.month, tm.year
                        # Check whether the file is present in the filesystem; if
                        # it doesn't, enter a while loop which periodically checks
                        # whether the file exists. We specify a maximum wait time
                        # in order to prevent the model hanging if the file never
                        # materialises.
                        exists = os.path.exists(gwFileNC)
                        max_wait_time = 60
                        wait_time = 0.1
                        total_wait_time = 0
                        while exists is False and total_wait_time <= max_wait_time:
                            time.sleep(wait_time)
                            exists = os.path.exists(gwFileNC)
                            total_wait_time += wait_time
                        if not exists:
                            msg = "groundwater file doesn't exist and maximum wait time exceeded"
                            raise ModelError(msg)

                        self.zGW = hm.open_hmdataarray(
                            self.gwFileNC,
                            self.config.WATER_TABLE['groundwaterVarName'],
                            self.domain
                        )
                        # zGW = file_handling.netcdf_to_arrayWithoutTime(
                        #     gwFileNC,
                        #     self._configuration.WATER_TABLE['groundwaterVarName'],
                        #     cloneMapFileName = self.cloneMapFileName
                        # )
                        # self.zGW = zGW[self.landmask]                        
                else:
                    self.zGW = hm.open_hmdataarray(
                        self.gwFileNC,
                        self.config.WATER_TABLE['groundwaterVarName'],
                        self.domain
                    )
                    # zGW = file_handling.netcdf_to_array(
                    #     self.gwFileNC,
                    #     self._configuration.WATER_TABLE['groundwaterVarName'],
                    #     str(currTimeStep.fulldate),
                    #     cloneMapFileName = self.cloneMapFileName
                    # )
                    # self.zGW = zGW[self.landmask]                    
            else:
                self.zGW = hm.open_hmdataarray(
                    self.gwFileNC,
                    self.config.WATER_TABLE['groudnwaterVarName'],
                    self.domain
                )
                # zGW = file_handling.netcdf_to_arrayWithoutTime(
                #     gwFileNC,
                #     self._configuration.WATER_TABLE['groundwaterVarName'],
                #     cloneMapFileName = self.cloneMapFileName,
                #     LatitudeLongitude = True)    
                # self.zGW = zGW[self.landmask]
                    
    def dynamic(self):
        self.read()

# def read_params(fn):
#     with open(fn) as f:
#         content = f.read().splitlines()
#     # remove commented lines
#     content = [x for x in content if re.search('^(?!%%).*', x)]
#     content = [re.split('\s*:\s*', x) for x in content]
#     params = {}
#     for x in content:
#         if len(x) > 1:
#             nm = x[0]
#             val = x[1]
#             params[nm] = val
#     return params
        
# class GroundwaterPoint(object):

#     def __init__(self, Groundwater_variable):
#         self._configuration = Groundwater_variable._configuration
#         self._modelTime = Groundwater_variable._modelTime
#         # self.cloneMapFileName = Groundwater_variable.cloneMapFileName
#         # self.landmask = Groundwater_variable.landmask

#     def initial(self):
#         # N.B. the first two values (excluding comments) define whether
#         # there is a groundwater table, and whether it's constant or
#         # variable; hence 'skiprows=2'
#         groundwater_file = self.var._configuration.WATER_TABLE['groundwaterFile']
#         with open(groundwater_file) as f:
#             content = f.read().splitlines()
#         # get the index of the first date match (this feels a bit hacky)
#         first_date = [i for i, val in enumerate(content) if re.search('^[0-9]{2}/[0-9]{2}/[0-9]{4}', val)]
#         if len(first_date) > 0:
#             first_date = [0]
            
#         # remove commented lines
#         content = [x for x in content if re.search('^(?!%).*', x)]
#         self.WaterTable = False
#         self.VariableWaterTable = False
#         if len(content) >= 2:            
#             if content[0] == 'Y' :
#                 self.WaterTable = True
#                 if content[1] == 'Variable':
#                     self.var.VariableWaterTable = True

#         if self.var.WaterTable:
#             d = pd.read_csv(
#                 self.var._configuration.WATER_TABLE['groundwaterFile'],
#                 delimiter='\s+|\t',
#                 header=None,
#                 names=['Date','zGW'],
#                 skiprows=first_date,
#                 comment="%",
#                 engine='python'
#             )
#             if self.var.VariableWaterTable:
#                 # UNTESTED
#                 dates = [datetime.datetime.strptime(date, '%d(/|-)%m(/|-)%Y') for date in d['Date']]
#                 zgw = list(d['zGW'])
#                 start_date = datetime.datetime(
#                     self._modelTime.startTime.year,
#                     self._modelTime.startTime.month,
#                     self._modelTime.startTime.day
#                 )
#                 end_date = datetime.datetime(
#                     self._modelTime.endTime.year,
#                     self._modelTime.month,
#                     self._modelTime.day
#                 )
#                 if start_date < dates[0]:
#                     dates = dates.insert(0, start_date)
#                     zgw = vals.insert(0, np.nan)
                    
#                 if end_date > dates[-1]:
#                     dates.append(end_date)
#                     zgw.append(np.nan)
#                 d = pd.DataFrame({'Date' : dates, 'zGW' : zgw})
#                 d.index = d['Date']
#                 del d['Date']
#                 d = d.resample('D').interpolate().ffill().bfill()                
#                 self.var.WaterTableDateFrame = d
#                 self.var.zGW = self.var.WaterTableDataFrame.loc[self.var._modelTime.startDate]
#             else:
#                 try:
#                     self.var.zGW = d['zGW'][0]
#                 except:
#                     self.var.zGW = 999

#     def read(self):
#         if self.var.VariableWaterTable:
#             try:
#                 self.var.zGW = self.var.WaterTableDataFrame.loc[self.var._modelTime.currTime]
#             except ValueError:
#                 pass
                    
#     def dynamic(self):
#         self.read()
        

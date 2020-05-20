#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar
import warnings
import scipy.interpolate as interpolate

# from hm import file_handling
from hm.api import open_hmdataarray

class InitialCondition(object):
    """Class to represent the initial condition of an AquaCrop run. 
    Although not yet implemented, this class should include a method 
    to read the model state from a dump netcdf, to enable a "warm" 
    start.
    """
    def __init__(self, InitialCondition_variable):
        self.var = InitialCondition_variable
        self.set_initial_condition_type()
        self.set_initial_condition_interp_method()
        
    def set_initial_condition_interp_method(self):
        self.var.initial_condition_interp_method = \
            self.var.config.INITIAL_WATER_CONTENT['interp_method'].upper()
        
    def set_initial_condition_type(self):
        self.var.initial_condition_type = \
            self.var.config.INITIAL_WATER_CONTENT['type'].upper()
        # self.var.initialConditionNC = None
        # self.var.initialConditionPercent = None
        # self.var.initialConditionProperty = None
        if self.var.initial_condition_type == 'FILE':
            self.var.initialConditionNC = \
                str(self.var.config.INITIAL_WATER_CONTENT['filename'])
        elif self.var.initial_condition_type == 'PERCENT':
            self.var.initialConditionPercent = \
                float(self.var.config.INITIAL_WATER_CONTENT['percent'])
        elif self.var.initial_condition_type == 'PROPERTY':
            self.var.initialConditionProperty = \
                str(self.var.config.INITIAL_WATER_CONTENT['property'])
            
    def interpolate_initial_condition_to_compartments_by_depth(self):
        zBot = np.cumsum(self.var.dz)
        zTop = zBot - self.var.dz
        zMid = (zTop + zBot) / 2
        self.var.th_init.interp(
            depth=zMid,
            method='linear',
            kwargs={'fill_value' : (self.var.th_init._data[0,...], self.var.th_init._data[-1,...])}
        )
    
    def interpolate_initial_condition_to_compartments_by_layer(self):
        zBot = np.cumsum(self.var.dz)
        zTop = zBot - self.var.dz
        zMid = (zTop + zBot) / 2
        self.var.th_init.select(depth=zMid, method='nearest')
    
    def interpolate_initial_condition_to_compartments(self):
        if self.var.initial_condition_interp_method == 'DEPTH':
            self.interpolate_initial_condition_to_compartments_by_depth()
        else:
            self.interpolate_initial_condition_to_compartments_by_layer()

    def set_initial_condition_from_file(self):
        self.var.th_init = open_hmdataarray(
            self.var.config.INITIAL_WATER_CONTENT['filename'],
            'th',
            self.var.domain,
            self.var.config.INITIAL_WATER_CONTENT['is_1d'],
            self.var.config.INITIAL_WATER_CONTENT['xy_dimname'],
            skip=['depth']
        )
        self.interpolate_initial_condition_to_compartments()
        self.var.th = np.broadcast_to(
            self.var.th_init.values[None,None,...],
            (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.domain.nxy)
        ).copy()
        
    def set_initial_condition_from_percent(self):
        self.var.th = (self.var.th_wilt_comp + ((self.var.initialConditionPercent / 100.) * (self.var.th_fc_comp - self.var.th_wilt_comp)))

    def set_initial_condition_from_property(self):
        if self.var.initialConditionProperty == 'FC':
            self.var.th = self.var.th_fc_comp.copy()
        elif self.var.initialConditionProperty == 'WP':
            self.var.th = self.var.th_wilt_comp.copy()
        elif self.var.initialConditionProperty == 'SAT':
            self.var.th = self.var.th_sat_comp.copy()

    def initial(self):
        if self.var.initial_condition_type == 'FILE':
            self.set_initial_condition_from_file()
        elif self.var.initial_condition_type == 'PERCENT':
            self.set_initial_condition_from_percent()
        elif self.var.initial_condition_type == 'PROPERTY':
            self.set_initial_condition_from_property()
        else:
            pass
        
    def dynamic(self):
        # Condition to identify crops which are not being grown or crops which
        # have only just finished being grown. The water content of crops
        # meeting this condition is used to compute the area-weighted initial
        # condition
        if np.any(self.var.GrowingSeasonDayOne):
            cond1 = np.logical_not(self.var.GrowingSeasonIndex) | self.var.GrowingSeasonDayOne
            cond1 = np.broadcast_to(cond1[:,:,None,:], self.var.th.shape)
            th = np.copy(self.var.th)
            th[(np.logical_not(cond1))] = np.nan

            # TEMPORARY FIX
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                th_ave = np.nanmean(th, axis=0) # average along farm dimension

            th_ave = np.broadcast_to(th_ave, (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.domain.nxy))
            # th_ave = th_ave[None,:,:] * np.ones((self.var.nCrop))[:,None,None]
            cond2 = np.broadcast_to(self.var.GrowingSeasonDayOne[:,:,None,:], self.var.th.shape)
            self.var.th[cond2] = th_ave[cond2]

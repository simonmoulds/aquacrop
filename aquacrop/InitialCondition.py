#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar
import warnings
import scipy.interpolate as interpolate

from hm import file_handling

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
            str(self.var._configuration.INITIAL_WATER_CONTENT['initialConditionInterpMethod']).upper()
        
    def set_initial_condition_type(self):
        self.var.initial_condition_type = \
            str(self.var._configuration.INITIAL_WATER_CONTENT['initialConditionType']).upper()
        self.var.initialConditionNC = None
        self.var.initialConditionPercent = None
        self.var.initialConditionProperty = None
        if self.var.initial_condition_type == 'FILE':
            self.var.initialConditionNC = \
                str(self.var._configuration.INITIAL_WATER_CONTENT['initialConditionNC'])
        elif self.var.initial_condition_type == 'PERCENT':
            self.var.initialConditionPercent = \
                float(self.var._configuration.INITIAL_WATER_CONTENT['initialConditionPercent'])
        elif self.var.initial_condition_type == 'PROPERTY':
            self.var.initialConditionProperty = \
                str(self.var._configuration.INITIAL_WATER_CONTENT['initialConditionProperty'])
            
    def interpolate_initial_condition_to_compartments_by_depth(self, th):
        depths = file_handling.get_dimension_variable(
            self.var.initialConditionNC, 'depth'
        )
        zBot = np.cumsum(self.var.dz)
        zTop = zBot - self.var.dz
        zMid = (zTop + zBot) / 2
        
        # add zero point
        if depths[0] > 0:
            depths = np.concatenate([[0.], depths])
            th = np.concatenate([th[0,...][None,:], th], axis=0)

        if depths[-1] < zBot[-1]:
            depths = np.concatenate([depths, [zBot[-1]]])
            th = np.concatenate([th, th[-1,...][None,:]], axis=0)

        # now interpolate to compartments
        f_thini = interpolate.interp1d(depths, th, axis=0, bounds_error=False, fill_value=(th[0,...], th[-1,...]))
        th = f_thini(zMid)
        return th
    
    def interpolate_initial_condition_to_compartments_by_layer(self, th):
        th = th[self.var.layerIndex,:,:]
        return th
    
    def interpolate_initial_condition_to_compartments(self, th):
        if self.var.initial_condition_interp_method == 'DEPTH':
            th = self.interpolate_initial_condition_to_compartments_by_depth(th)
        else:
            th = self.interpolate_initial_condition_to_compartments_by_layer(th)
        return th

    def set_initial_condition_from_file(self):
        # TODO: use config to set initialConditionVarName, initialConditionDepthVarName
        th = file_handling.netcdf_to_arrayWithoutTime(
            self.var.initialConditionNC,
            'th',
            cloneMapFileName=self.var.cloneMapFileName)
        th = self.interpolate_initial_condition_to_compartments(th)
        th = th[self.var.landmask_comp].reshape(self.var.nComp, self.var.nCell)
        self.var.th = np.broadcast_to(
            th[None,None,...],
            (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell)
        ).copy()
        
    def set_initial_condition_from_percent(self):
        # comp_index = np.unique(self.var.layerIndex, return_index=True)[1]        
        self.var.th = (
            self.var.th_wilt_comp
            + ((self.var.initialConditionPercent / 100.)
               * (self.var.th_fc_comp - self.var.th_wilt_comp))
        )

    def set_initial_condition_from_property(self):
        # comp_index = np.unique(self.var.layerIndex, return_index=True)[1]                
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
        
        # # # Define initial water contents
        # # self.var.initialConditionNC = self.var._configuration.INITIAL_WATER_CONTENT['initialConditionNC']
        # # th = file_handling.netcdf_to_arrayWithoutTime(
        # #     self.var.initialConditionNC,
        # #     'th',
        # #     cloneMapFileName=self.var.cloneMapFileName)
        
        # # init_cond_interp_method = self.var._configuration.INITIAL_WATER_CONTENT['InterpMethod']
        # # init_cond_type = self.var._configuration.globalOptions['initialConditionType']

        # if self.var.initial_condition_interp_method.lower() == 'depth':
        #     depths = file_handling.get_dimension_variable(
        #         self.var.initialConditionNC,'depth')
        #     # depths = file_handling.netcdfDim2NumPy(
        #     #     self.var.initialConditionNC,'depth')
        #     zBot = np.cumsum(self.var.dz)
        #     zTop = zBot - self.var.dz
        #     zMid = (zTop + zBot) / 2

        #     # add zero point
        #     if depths[0] > 0:
        #         depths = np.concatenate([[0.], depths])
        #         th = np.concatenate([th[0,...][None,:], th], axis=0)

        #     if depths[-1] < zBot[-1]:
        #         depths = np.concatenate([depths, [zBot[-1]]])
        #         th = np.concatenate([th, th[-1,...][None,:]], axis=0)

        #     # now interpolate to compartments
        #     f_thini = interpolate.interp1d(depths, th, axis=0, bounds_error=False, fill_value=(th[0,...], th[-1,...]))
        #     th = f_thini(zMid)
        # else:
        #     th = th[self.var.layerIndex,:,:]
       
        # # NB original Matlab code also allows users to supply initial condition
        # # as field capacity, wilting point or saturation. We do not explicitly
        # # include this option but of course it is possible to do this by
        # # supplying a netCDF file containing the values. However, note that in
        # # the original version if field capacity is specified and groundwater
        # # table is present the initial water content is set to th_fc_adj once
        # # this variable has been computed.

        # th = th[self.var.landmask_comp].reshape(self.var.nComp, self.var.nCell)
        # self.var.th = np.broadcast_to(th[None,None,...], (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell)).copy()

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

            th_ave = np.broadcast_to(th_ave, (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell))
            # th_ave = th_ave[None,:,:] * np.ones((self.var.nCrop))[:,None,None]
            cond2 = np.broadcast_to(self.var.GrowingSeasonDayOne[:,:,None,:], self.var.th.shape)
            self.var.th[cond2] = th_ave[cond2]

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from hm import file_handling
import logging
logger = logging.getLogger(__name__)

class Irrigation(object):
    """Class to represent irrigation activities"""

    def __init__(self, Irrigation_variable):
        self.var = Irrigation_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.Irr = np.copy(arr_zeros)
        self.var.IrrCum = np.copy(arr_zeros)
        self.var.IrrNetCum = np.copy(arr_zeros)
        self.irrigate_soil_moisture_threshold = (self.var.IrrMethod == 1)
        self.irrigate_fixed_interval = (self.var.IrrMethod == 2)
        self.irrigate_from_schedule = (self.var.IrrMethod == 3)
        self.irrigate_net = (self.var.IrrMethod == 4)
        
    def reset_initial_conditions(self):
        self.var.IrrCum[self.var.GrowingSeasonDayOne] = 0
        self.var.IrrNetCum[self.var.GrowingSeasonDayOne] = 0

    def adjust_root_zone_depletion(self):
        rootdepth = np.maximum(self.var.Zmin, self.var.Zroot)
        AbvFc = ((self.var.thRZ_Act - self.var.thRZ_Fc) * 1000 * rootdepth)
        AbvFc = np.clip(AbvFc, 0., None)
        WCadj = self.var.ETpot - self.var.weather.precipitation + self.var.Runoff - AbvFc
        Dr = self.var.Dr + WCadj
        Dr = np.clip(Dr, 0., None)
        return Dr
    
    def compute_irrigation_depth_satoil_moisture_threshold(self, WCadj):
        # If irrigation is based on soil moisture, get the soil moisture
        # target for the current growth stage and determine threshold to
        # initiate irrigation
        I,J,K = np.ogrid[:self.var.nFarm,:self.var.nCrop,:self.var.nCell]
        growth_stage_index = self.var.GrowthStage.astype(int) - 1
        SMT = np.concatenate((self.var.SMT1[None,:],
                              self.var.SMT2[None,:],
                              self.var.SMT3[None,:],
                              self.var.SMT4[None,:]), axis=0)
        SMT = SMT[growth_stage_index,I,J,K]
        IrrThr = np.round(((1. - SMT / 100.) * self.var.TAW), 3)
        IrrReq = self.adjust_root_zone_depletion()
        EffAdj = ((100. - self.var.AppEff) + 100.) / 100.  # ???
        IrrReq *= EffAdj
        irrigate = self.var.GrowingSeasonIndex & self.irrigate_soil_moisture_threshold & (Dr > IrrThr)
        self.var.Irr[irrigate] = np.clip(IrrReq, 0, self.var.MaxIrr)[irrigate]

    def compute_irrigation_depth_fixed_interval(self):
        """Irrigation depth using fixed interval method."""
        IrrReq = self.adjust_root_zone_depletion()
        EffAdj = ((100. - self.var.AppEff) + 100.) / 100.
        IrrReq *= EffAdj
        IrrReq = np.clip(IrrReq, 0., self.var.MaxIrr)
        nDays = self.var.DAP - 1
        irrigate = self.var.GrowingSeasonIndex & self.irrigate_fixed_interval & ((nDays % self.var.IrrInterval) == 0)
        self.var.Irr[irrigate] = IrrReq[irrigate]

    def compute_irrigation_depth_satchedule(self):
        # If irrigation is based on a pre-defined schedule then the irrigation
        # requirement for each crop is read from a netCDF file. Note that if
        # the option 'irrScheduleFileNC' is None, then nothing will be imported
        # and the irrigation requirement will be zero
        IrrReq = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        if self.var.irrScheduleFileNC != None:            
            IrrReq = file_handling.netcdf_to_array(
                self.var.irrScheduleFileNC,
                "irrigation_depth",
                str(self.var._modelTime.fulldate), 
                cloneMapFileName = self.var.cloneMapFileName)
            IrrReq = IrrReq[self.var.landmask_crop].reshape(self.var.nCrop,self.var.nCell)
            
        irrigate = self.var.GrowingSeasonIndex & self.irrigate_from_schedule
        self.var.Irr[irrigate] = IrrReq[irrigate]

    def compute_irrigation_depth_net(self):        
        # Note that if irrigation is based on net irrigation then it is
        # performed after calculation of transpiration. A dummy method is
        # included here in order to provide this explanation.
        pass
    
    def compute_irrigation_depth(self):
        self.var.Irr[:] = 0.
        if np.any(self.var.GrowingSeasonIndex):
            if np.any(self.irrigate_soil_moisture_threshold):
                self.compute_irrigation_depth_satoil_moisture_threshold()
            if np.any(self.irrigate_fixed_interval):
                self.compute_irrigation_depth_fixed_interval()                    
            if np.any(self.irrigate_from_schedule):
                self.compute_irrigation_depth_satchedule()
            if np.any(self.irrigate_net):
                self.compute_irrigation_depth_net()
        self.var.IrrCum += self.var.Irr
        self.var.IrrCum[np.logical_not(self.var.GrowingSeasonIndex)] = 0
    
    def dynamic(self):
        """Function to get irrigation depth for the current day"""
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()
        self.compute_irrigation_depth()

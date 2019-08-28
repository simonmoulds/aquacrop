#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

class Inflow(object):
    """Class to check for presence of a groundwater table and, if 
    present, to adjust compartment water contents and field 
    capacities accordingly.
    """
    def __init__(self, GroundwaterInflow_variable):
        self.var = GroundwaterInflow_variable

    def initial(self):
        self.GwIn = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        
    def dynamic(self):
        """Function to calculate capillary rise in the presence of a 
        shallow groundwater table
        """

        zGW_comp = np.broadcast_to(self.var.groundwater.zGW[None,None,None,:], (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell))        
        GwIn = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))

        # Water table in soil profile: calculate horizontal inflow; get
        # groundwater table elevation on current day
        zBot = np.cumsum(self.var.dz_xy, axis=2)  # sum along comp axis
        zTop = zBot - self.var.dz_xy
        zMid = (zTop + zBot) / 2

        # For compartments below water table, set to saturation
        dth = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell))

        cond1 = (np.broadcast_to(self.var.WTinSoil[:,:,None,:], self.var.th.shape)
                 & np.greater_equal(zMid, zGW_comp)# np.broadcast_to(self.var.zGW[:,None,:], self.var.th.shape))
                 & np.less(self.var.th, self.var.th_sat_comp))

        # Update water content
        dth[cond1] = (self.var.th_sat_comp - self.var.th)[cond1]
        self.var.th[cond1] = self.var.th_sat_comp[cond1]

        # Update groundwater inflow
        GwIn_comp = dth * 1000 * self.var.dz_xy
        self.var.GwIn = np.sum(GwIn_comp, axis=2)  # sum along comp axis

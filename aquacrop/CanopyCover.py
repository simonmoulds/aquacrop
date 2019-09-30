#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class CanopyCover(object):
    """Class to represent canopy growth/decline."""
    def __init__(self, CanopyCover_variable):
        """Create a CanopyCover object by providing an AquaCrop 
        object.
        """
        self.var = CanopyCover_variable

    def initial(self):
        """Initialize the NumPy data structures modified by the class 
        instance.
        """
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.tEarlySen = np.copy(arr_zeros)
        self.var.CC = np.copy(arr_zeros)
        self.var.CCadj = np.copy(arr_zeros)
        self.var.CC_NS = np.copy(arr_zeros)
        self.var.CCadj_NS = np.copy(arr_zeros)
        self.var.CCxAct = np.copy(arr_zeros)
        self.var.CCxAct_NS = np.copy(arr_zeros)
        self.var.CCxW = np.copy(arr_zeros)
        self.var.CCxW_NS = np.copy(arr_zeros)
        self.var.CCxEarlySen = np.copy(arr_zeros)
        self.var.CCprev = np.copy(arr_zeros)
        self.var.PrematSenes = np.copy(arr_zeros.astype(bool))        
        self.var.CropDead = np.copy(arr_zeros.astype(bool))        
        self.var.CC0adj = np.copy(arr_zeros)

    def reset_initial_conditions(self):
        """Reset the value of various variables at the beginning of 
        the growing season.
        """
        self.var.tEarlySen[self.var.GrowingSeasonDayOne] = 0
        self.var.CC[self.var.GrowingSeasonDayOne] = 0
        self.var.CCadj[self.var.GrowingSeasonDayOne] = 0
        self.var.CC_NS[self.var.GrowingSeasonDayOne] = 0
        self.var.CCadj_NS[self.var.GrowingSeasonDayOne] = 0
        self.var.CCxAct[self.var.GrowingSeasonDayOne] = 0
        self.var.CCxAct_NS[self.var.GrowingSeasonDayOne] = 0
        self.var.CCxW[self.var.GrowingSeasonDayOne] = 0
        self.var.CCxW_NS[self.var.GrowingSeasonDayOne] = 0
        self.var.CCxEarlySen[self.var.GrowingSeasonDayOne] = 0
        self.var.CCprev[self.var.GrowingSeasonDayOne] = 0
        self.var.PrematSenes[self.var.GrowingSeasonDayOne] = False
        self.var.CropDead[self.var.GrowingSeasonDayOne] = False
        self.var.CC0adj[self.var.GrowingSeasonDayOne] = self.var.CC0[self.var.GrowingSeasonDayOne]
        
    def dynamic(self):
        """Update CanopyCover object for the current time step."""
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()            

        aquacrop_fc.canopy_cover_w.update_canopy_cover_w(
            np.asfortranarray(self.var.CC),
            np.asfortranarray(self.var.CCprev),
            np.asfortranarray(self.var.CCadj),
            np.asfortranarray(self.var.CC_NS),
            np.asfortranarray(self.var.CCadj_NS),
            np.asfortranarray(self.var.CCxW),
            np.asfortranarray(self.var.CCxAct),
            np.asfortranarray(self.var.CCxW_NS),
            np.asfortranarray(self.var.CCxAct_NS),
            np.asfortranarray(self.var.CC0adj),
            np.asfortranarray(self.var.CCxEarlySen),
            np.asfortranarray(self.var.tEarlySen),
            np.asfortranarray(np.int32(self.var.PrematSenes)),
            np.asfortranarray(np.int32(self.var.CropDead)),
            np.asfortranarray(self.var.GDD),
            np.asfortranarray(self.var.GDDcum),
            np.asfortranarray(self.var.CC0),
            np.asfortranarray(self.var.CCx),
            np.asfortranarray(self.var.CGC),
            np.asfortranarray(self.var.CDC),
            np.asfortranarray(self.var.Emergence),
            np.asfortranarray(self.var.Maturity),
            np.asfortranarray(self.var.Senescence),
            np.asfortranarray(self.var.CanopyDevEnd),
            np.asfortranarray(self.var.Dr),
            np.asfortranarray(self.var.TAW),
            np.asfortranarray(self.var.weather.referencePotET),
            np.asfortranarray(self.var.ETadj),
            np.asfortranarray(self.var.p_up1),
            np.asfortranarray(self.var.p_up2),
            np.asfortranarray(self.var.p_up3),
            np.asfortranarray(self.var.p_up4),
            np.asfortranarray(self.var.p_lo1),
            np.asfortranarray(self.var.p_lo2),
            np.asfortranarray(self.var.p_lo3),
            np.asfortranarray(self.var.p_lo4),
            np.asfortranarray(self.var.fshape_w1),
            np.asfortranarray(self.var.fshape_w2),
            np.asfortranarray(self.var.fshape_w3),
            np.asfortranarray(self.var.fshape_w4),
            np.asfortranarray(self.var.GrowingSeasonIndex),
            int(self.var.CalendarType),
            np.asfortranarray(self.var.DAP),
            np.asfortranarray(self.var.DelayedCDs),
            np.asfortranarray(self.var.DelayedGDDs),
            int(self.var.nFarm),
            int(self.var.nCrop),
            int(self.var.nCell)
        )
            

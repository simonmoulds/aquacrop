#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class CanopyCover(object):
    def __init__(self, CanopyCover_variable):
        """Create a CanopyCover object by providing an AquaCrop 
        object.
        """
        self.var = CanopyCover_variable

    def initial(self):
        """Initialize the NumPy data structures modified by the class 
        instance.
        """
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
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
        self.var.PrematSenes = np.copy(arr_zeros.astype(np.int32))        
        self.var.CropDead = np.copy(arr_zeros.astype(np.int32))        
        self.var.CC0adj = np.copy(arr_zeros)
    
    def dynamic(self):
        """Update CanopyCover object for the current time step."""
        aquacrop_fc.canopy_cover_w.update_canopy_cover_w(
            self.var.CC.T,
            self.var.CCprev.T,
            self.var.CCadj.T,
            self.var.CC_NS.T,
            self.var.CCadj_NS.T,
            self.var.CCxW.T,
            self.var.CCxAct.T,
            self.var.CCxW_NS.T,
            self.var.CCxAct_NS.T,
            self.var.CC0adj.T,
            self.var.CCxEarlySen.T,
            self.var.tEarlySen.T,
            self.var.PrematSenes.T,
            self.var.CropDead.T,
            self.var.GDD.T,
            self.var.GDDcum.T,
            self.var.CC0.T,
            self.var.CCx.T,
            self.var.CGC.T,
            self.var.CDC.T,
            self.var.Emergence.T,
            self.var.Maturity.T,
            self.var.Senescence.T,
            self.var.CanopyDevEnd.T,
            self.var.Dr.T,
            self.var.TAW.T,
            self.var.model.etref.values.T,
            # self.var.weather.referencePotET.T,
            self.var.ETadj.T,
            self.var.p_up1.T,
            self.var.p_up2.T,
            self.var.p_up3.T,
            self.var.p_up4.T,
            self.var.p_lo1.T,
            self.var.p_lo2.T,
            self.var.p_lo3.T,
            self.var.p_lo4.T,
            self.var.fshape_w1.T,
            self.var.fshape_w2.T,
            self.var.fshape_w3.T,
            self.var.fshape_w4.T,
            self.var.GrowingSeasonIndex.T,
            self.var.GrowingSeasonDayOne.T,
            int(self.var.CalendarType),
            self.var.DAP.T,
            self.var.DelayedCDs.T,
            self.var.DelayedGDDs.T,
            int(self.var.nFarm),
            int(self.var.nCrop),
            int(self.var.domain.nxy)
        )

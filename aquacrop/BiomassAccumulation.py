#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class BiomassAccumulation(object):
    def __init__(self, BiomassAccumulation_variable):
        self.var = BiomassAccumulation_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.B = np.copy(arr_zeros)
        self.var.B_NS = np.copy(arr_zeros)

    def dynamic(self):
        """Function to calculate biomass accumulation"""
        aquacrop_fc.biomass_accumulation_w.update_biomass_accum_w(
            self.var.model.etref.values.T,
            # self.var.weather.referencePotET.T,
            self.var.TrAct.T,
            self.var.TrPot_NS.T,
            self.var.B.T,
            self.var.B_NS.T,
            self.var.BioTempStress.T,
            self.var.GDD.T,
            self.var.GDD_up.T,
            self.var.GDD_lo.T,
            self.var.PolHeatStress.T,
            self.var.model.tmax.values.T,
            # self.var.weather.tmax.T,
            self.var.Tmax_up.T,
            self.var.Tmax_lo.T,
            self.var.fshape_b.T,
            self.var.PolColdStress.T,
            self.var.model.tmin.values.T,
            # self.var.weather.tmin.T,
            self.var.Tmin_up.T,
            self.var.Tmin_lo.T,
            self.var.HI.T,
            self.var.PctLagPhase.T,
            self.var.YldFormCD.T,
            self.var.WP.T,
            self.var.WPy.T,
            self.var.fCO2.T,
            self.var.HIstartCD.T,
            self.var.DelayedCDs.T,
            self.var.DAP.T,
            self.var.CropType.T,
            self.var.Determinant.T,
            self.var.GrowingSeasonIndex.T,
            self.var.nFarm,
            self.var.nCrop,
            self.var.domain.nxy
        )

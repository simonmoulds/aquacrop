#!/usr/bin/env python
# -*- coding: utf-8 -*-

# NB: this class has not been tested!

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class CapillaryRise(object):
    def __init__(self, CapillaryRise_variable):
        self.var = CapillaryRise_variable

    def initial(self):
        self.var.CrTot = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))

    def dynamic(self):
        """Function to calculate capillary rise from a shallow 
        groundwater table
        """
        layer_ix = self.var.layerIndex + 1
        aquacrop_fc.capillary_rise_w.update_cap_rise_w(
            self.var.CrTot.T,
            self.var.th.T,
            self.var.th_wilt.T,
            self.var.th_fc.T,
            self.var.th_fc_adj.T,
            self.var.k_sat.T,
            self.var.aCR.T,
            self.var.bCR.T,
            self.var.fshape_cr.T,
            self.var.FluxOut.T,
            self.var.groundwater.WaterTable,
            self.var.groundwater.zGW,
            self.var.dz,
            self.var.dz_layer,
            layer_ix,
            self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nLayer, self.var.domain.nxy
            )

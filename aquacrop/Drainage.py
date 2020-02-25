#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class Drainage(object):
    def __init__(self, Drainage_variable):
        self.var = Drainage_variable

    def initial(self):
        self.var.FluxOut = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.domain.nxy))
        self.var.DeepPerc = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.Recharge = np.zeros((self.var.domain.nxy))

    def dynamic(self):
        """Function to redistribute stored soil water"""
        layer_ix = self.var.layerIndex + 1
        aquacrop_fc.drainage_w.update_drainage_w(
            self.var.th.T,
            self.var.DeepPerc.T,
            self.var.FluxOut.T,
            self.var.th_sat.T,
            self.var.th_fc.T,
            self.var.k_sat.T,
            self.var.tau.T,
            self.var.th_fc_adj.T,
            self.var.dz,
            self.var.dz_sum,
            layer_ix,
            self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nLayer, self.var.domain.nxy
            )

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .constants import sigma, alpha

class NetRadiation(object):
    def __init__(self, model):
        self.model = model
        self.alpha = 0.23 # albedo, 0.23 [-]
        self.sigma = 4.903e-9 # stephan boltzmann [W m-2 K-4]
        
    def initial(self):
        pass

    def compute_net_radiation(self):        
        Rso = np.maximum(
            0.1,
            ((0.75 + (2 * 0.00005)) * self.model.extraterrestrial_radiation)
        ) # clear sky solar radiation MJ d-1
        Rsin_MJ = 0.086400 * self.model.shortwave_radiation.values
        Rlnet_MJ = (
            - sigma
            * ((self.model.tmax.values ** 4 + self.model.tmin.values ** 4) / 2)
            * (0.34 - 0.14 * np.sqrt(np.maximum(0, (self.model.ea_mean / 1000))))
            * (1.35 * np.minimum(1, (Rsin_MJ / Rso)) - 0.35)
        )
        Rlnet_Watt = Rlnet_MJ / 0.086400
        self.model.net_radiation = np.maximum(
            0,
            ((1 - alpha) * self.model.shortwave_radiation.values + Rlnet_Watt)
        )
        
    def dynamic(self):
        self.compute_net_radiation()

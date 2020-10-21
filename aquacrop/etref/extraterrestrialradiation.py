#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class ExtraterrestrialRadiation(object):
    def __init__(self, model):
        self.model = model

    def initial(self):
        latitudes = self.model.domain.y
        if self.model.domain.is_2d:
            latitudes = np.broadcast_to(
                self.model.domain.y[:,None],
                (self.model.domain.ny, self.model.domain.nx)
            )
            
        self.latitudes = latitudes[self.model.domain.mask.values]
        
    def compute_extraterrestrial_radiation(self):
        """Compute extraterrestrial radiation (MJ m-2 d-1)"""
        LatRad = self.latitudes * np.pi / 180.0
        declin = (0.4093 * (np.sin(((2.0 * np.pi * self.model.time.doy) / 365.0) - 1.405)))
        arccosInput = (-(np.tan(LatRad)) * (np.tan(declin)))
        arccosInput = np.clip(arccosInput, -1, 1)
        sunangle = np.arccos(arccosInput)
        distsun = 1 + 0.033 * (np.cos((2 * np.pi * self.model.time.doy) / 365.0))
        self.model.extraterrestrial_radiation = (
            ((24 * 60 * 0.082) / np.pi)
            * distsun
            * (sunangle
               * (np.sin(LatRad))
               * (np.sin(declin))
               + (np.cos(LatRad))
               * (np.cos(declin))
               * (np.sin(sunangle)))
        )
        
    def dynamic(self):
        self.compute_extraterrestrial_radiation()

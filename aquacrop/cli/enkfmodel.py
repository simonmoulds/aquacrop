#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

from hm.dynamicmodel import HmEnKfModel

# To begin with we consider the simplest case, where the model is running for one point in space

class AqEnKfModel(HmEnKfModel):

    def setState(self):
        modelled_canopy_cover = np.array((self.model.CC[0,0,0],))
        return modelled_canopy_cover

    def setObservations(self):
        timestep = self.currentTimeStep()
        fn = 'obs' + str(timestep) + '.txt'
        print(fn)
        with open(fn) as f:
            obs_canopy_cover = [float(val) for val in f.read().split()]

        obs_canopy_cover = np.array([obs_canopy_cover,] * self.nrSamples()).transpose()
        covariance = np.random.random((1,1))
        # print(fn)
        # print(obs_canopy_cover)
        # print(covariance)
        self.setObservedMatrices(obs_canopy_cover, covariance)        
            
    def resume(self):
        pass
    

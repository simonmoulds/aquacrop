#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Evapotranspiration(object):
    """Class to represent combined soil evaporation and plant
    transpiration
    """
    def __init__(self, Evapotranspiration_variable):
        self.var = Evapotranspiration_variable
        
    def initial(self):        
        self.var.ETpot = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        
    def dynamic(self):
        self.var.ETpot = self.var.Epot + self.var.Tpot

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Evapotranspiration(object):
    def __init__(self, Evapotranspiration_variable):
        self.var = Evapotranspiration_variable
        
    def initial(self):        
        self.var.ETpot = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        
    def dynamic(self):
        self.var.ETpot = self.var.Epot + self.var.Tpot

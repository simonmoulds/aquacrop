#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime

from .SoilHydraulicParameters import SoilHydraulicParameters
from .SoilParameters import SoilParameters
from .CropArea import CropArea
from .CropParameters import CropParameters
from .FarmParameters import FarmParameters
from .IrrigationManagementParameters import IrrigationManagementParameters
from .FieldManagementParameters import FieldManagementParameters
    
class CoverFraction(object):
    def __init__(self, model):
        self.model = model        
    def initial(self):
        self.update_cover_fraction()        
    def update_cover_fraction(self):
        self.model.cover_fraction = np.ones_like(self.model.domain.mask)        
    def dynamic(self):
        self.update_cover_fraction()

class AquaCropParameters(object):
    def __init__(self, model):
        self.soil_hydraulic_parameters_module = SoilHydraulicParameters(model)
        self.soil_parameters_module = SoilParameters(model)
        self.cover_fraction_module = CoverFraction(model)        
        self.farm_parameters_module = FarmParameters(model)
        self.crop_parameters_module = CropParameters(model)
        self.crop_area_module = CropArea(model)
        self.field_mgmt_parameters_module = FieldManagementParameters(model)
        self.irrigation_parameters_module = IrrigationManagementParameters(model)

    def initial(self):
        self.soil_hydraulic_parameters_module.initial()
        self.soil_parameters_module.initial()
        self.cover_fraction_module.initial()
        self.farm_parameters_module.initial()
        self.crop_parameters_module.initial()
        self.crop_area_module.initial()
        self.field_mgmt_parameters_module.initial()
        self.irrigation_parameters_module.initial()

    def dynamic(self):
        self.soil_hydraulic_parameters_module.dynamic()
        self.soil_parameters_module.dynamic()
        self.cover_fraction_module.dynamic()
        self.farm_parameters_module.dynamic()
        self.crop_parameters_module.dynamic()
        self.crop_area_module.dynamic()
        self.field_mgmt_parameters_module.dynamic()
        self.irrigation_parameters_module.dynamic()

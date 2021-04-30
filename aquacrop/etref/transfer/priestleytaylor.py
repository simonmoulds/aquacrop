#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from hm.model import Model

from .extraterrestrialradiation import ExtraterrestrialRadiation
from .inputdata import (LongwaveRadiation,
                        ShortwaveRadiation,
                        RelativeHumidity,
                        SpecificHumidity,
                        SurfacePressure,
                        Temperature,
                        Wind)
from .netradiation import NetRadiation
# from .relativehumidity import RelativeHumidity
# from .specifichumidity import SpecificHumidity
# from .surfacepressure import SurfacePressure
# from .temperature import Temperature
from .vapourpressure import VapourPressure
# from .wind import Wind

import logging
logger = logging.getLogger(__name__)

class PriestleyTaylor(Model):
    
    def __init__(self, configuration, modelTime, initialState):        
        super(PenmanMonteith, self).__init__(
            configuration,
            modelTime,
            initialState)
        self.temperature_module = Temperature(self)
        self.shortwave_radiation_module = ShortwaveRadiation(self)
        self.extraterrestrial_radiation_module = ExtraterrestrialRadiation(self)
        self.specific_humidity_module = SpecificHumidity(self)
        self.relative_humidity_module = RelativeHumidity(self)
        self.surface_pressure_module = SurfacePressure(self)
        self.vapour_pressure_deficit_module = VapourPressureDeficit(self)
        self.net_radiation_module = NetRadiation(self)

    def initial(self):
        self.shortwave_radiation_module.initial()
        self.extraterrestrial_radiation_module.initial()
        self.relative_humidity_module.initial()
        self.specific_humidity_module.initial()
        self.surface_pressure_module.initial()
        self.vapour_pressure_deficit_module.initial()        
        self.net_radiation_module.initial()        
        self.temperature_module.initial()

    def priestley_taylor(self):
        """ http://agsys.cra-cin.it/tools/evapotranspiration/help/Priestley-Taylor.html """
        cp = 0.001013   # specific heat of air 1013 [MJ kg-1 K-1]
        a = 1.26        # Priestley-Taylor coefficient [-]
        eps = 0.622     # ratio of molecular weight of water to dry air [-]        
        latent_heat = 2.501 - (0.002361 * (self.tmean - 273.15)) # MJ kg-1        
        slope_exp   = (17.27 * (self.tmean - 273.15)) / ((self.tmean - 273.15) + 237.3)
        slope_div   = ((self.tmean - 273.15) + 237.3) ** 2
        delta       = 4098 * (0.6108 * (np.exp(slope_exp))) / slope_div # kPa K-1
        gamma  = cp * (self.surface_pressure / 1000) / (latent_heat * eps) # psychrometric constant (kPa K-1)
        PETmm = a * (1 / latent_heat) * ((delta * self.net_radiation * 0.086400) /  (delta + gamma))
        self.ETref = PETmm.copy()
    
    def dynamic(self):
        logger.info("Reading forcings for time %s", self._modelTime)
        self.temperature_module.dynamic()
        self.shortwave_radiation_module.dynamic()
        self.extraterrestrial_radiation_module.dynamic()
        self.relative_humidity_module.dynamic()
        self.specific_humidity_module.dynamic()
        self.surface_pressure_module.dynamic()
        self.vapour_pressure_deficit_module.dynamic()
        self.net_radiation_module.dynamic()        
        self.priestley_taylor()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import numpy as np
import netCDF4 as nc
from hm import file_handling

class SoilHydraulicParameters(object):
    def __init__(self, SoilHydraulicParameters_variable, config_section_name):
        self.var = SoilHydraulicParameters_variable
        self.var.dz = self.var._configuration.SOIL_HYDRAULIC_PARAMETERS['dzSoilCompartment']
        self.var.dz_sum = np.cumsum(self.var.dz)
        self.var.dz_layer = self.var._configuration.SOIL_HYDRAULIC_PARAMETERS['dzSoilLayer']
        self.var.nComp = len(self.var.dz)
        self.var.nLayer = len(self.var.dz_layer)
        
    def initial(self):
        self.var.soilHydraulicParametersNC = self.var._configuration.SOIL_HYDRAULIC_PARAMETERS['soilHydraulicParametersNC']
        self.var.dz_xy = np.broadcast_to(self.var.dz[None,None,:,None], (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell))
        self.var.dz_sum_xy = np.broadcast_to(self.var.dz_sum[None,None,:,None], (self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell))
        self.var.landmask_comp = np.broadcast_to(
            self.var.landmask[None,...],
            (self.var.nComp, self.var.nLat, self.var.nLon)
            )
        self.var.landmask_layer = np.broadcast_to(
            self.var.landmask[None,...],
            (self.var.nLayer, self.var.nLat, self.var.nLon)
            )
        self.read_soil_hydraulic_parameters()
        self.compute_additional_soil_hydraulic_parameters()
        self.compute_capillary_rise_parameters()
        
    def read_soil_hydraulic_parameters(self):        
        soil_hydraulic_parameters = {
            'k_sat'  : self.var._configuration.SOIL_HYDRAULIC_PARAMETERS['saturatedHydraulicConductivityVarName'],
            'th_sat' : self.var._configuration.SOIL_HYDRAULIC_PARAMETERS['saturatedVolumetricWaterContentVarName'],
            'th_fc'  : self.var._configuration.SOIL_HYDRAULIC_PARAMETERS['fieldCapacityVolumetricWaterContentVarName'],
            'th_wilt': self.var._configuration.SOIL_HYDRAULIC_PARAMETERS['wiltingPointVolumetricWaterContentVarName']
        }
        for param, var_name in soil_hydraulic_parameters.items():
            d = file_handling.netcdf_to_arrayWithoutTime(
                self.var.soilHydraulicParametersNC,
                var_name,
                cloneMapFileName=self.var.cloneMapFileName
            )  # TODO: check dimensions are as expected
            d = d[self.var.landmask_layer].reshape(self.var.nLayer,self.var.nCell)
            vars(self.var)[param] = np.broadcast_to(
                d[None,None,:,:],
                (self.var.nFarm, self.var.nCrop, self.var.nLayer, self.var.nCell)
            )
            
    def compute_additional_soil_hydraulic_parameters(self):
        zBot = np.cumsum(self.var.dz)
        zTop = zBot - self.var.dz
        zMid = (zTop + zBot) / 2
        dz_layer_bot = np.cumsum(self.var.dz_layer)
        dz_layer_top = dz_layer_bot - self.var.dz_layer        
        self.var.layerIndex = np.sum(((zMid[:,None] * np.ones((self.var.nLayer))[None,:]) > dz_layer_top), axis=1) - 1
        
        # The following is adapted from AOS_ComputeVariables.m, lines 129-139
        # "Calculate drainage characteristic (tau)
        self.var.tau = 0.0866 * (self.var.k_sat ** 0.35)
        self.var.tau = np.round(self.var.tau * 100) / 100
        self.var.tau[self.var.tau > 1] = 1
        self.var.tau[self.var.tau < 0] = 0

        # The following is adapted from AOS_ComputeVariables.m, lines 25
        self.var.th_dry = self.var.th_wilt / 2

        # transform certain soil properties to (ncrop, ncomp, ncell)
        soil_params = ['th_sat','th_fc','th_wilt','th_dry','k_sat','tau']
        for nm in soil_params:
            newnm = nm + '_comp'
            vars(self.var)[newnm] = vars(self.var)[nm][...,self.var.layerIndex,:]
            
    def compute_capillary_rise_parameters(self):
        # Function adapted from AOS_ComputeVariables.m, lines 60-127

        self.var.aCR = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nLayer, self.var.nCell))
        self.var.bCR = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nLayer, self.var.nCell))

        # "Sandy soil class"
        cond1 = (self.var.th_wilt >= 0.04) & (self.var.th_wilt <= 0.15) & (self.var.th_fc >= 0.09) & (self.var.th_fc <= 0.28) & (self.var.th_sat >= 0.32) & (self.var.th_sat <= 0.51)
        cond11 = (cond1 & (self.var.k_sat >= 200) & (self.var.k_sat <= 2000))
        cond12 = (cond1 & (self.var.k_sat < 200))
        cond13 = (cond1 & (self.var.k_sat > 2000))
        self.var.aCR[cond11] = (-0.3112 - (self.var.k_sat * (10 ** -5)))[cond11]
        self.var.bCR[cond11] = (-1.4936 + (0.2416 * np.log(self.var.k_sat)))[cond11]
        self.var.aCR[cond12] = (-0.3112 - (200 * (10 ** -5)))
        self.var.bCR[cond12] = (-1.4936 + (0.2416 * np.log(200)))
        self.var.aCR[cond13] = (-0.3112 - (2000 * (10 ** -5)))
        self.var.bCR[cond13] = (-1.4936 + (0.2416 * np.log(2000)))

        # "Loamy soil class"
        cond2 = (self.var.th_wilt >= 0.06) & (self.var.th_wilt <= 0.20) & (self.var.th_fc >= 0.23) & (self.var.th_fc <= 0.42) & (self.var.th_sat >= 0.42) & (self.var.th_sat <= 0.55)
        cond21 = (cond2 & (self.var.k_sat >= 100) & (self.var.k_sat <= 750))
        cond22 = (cond2 & (self.var.k_sat < 100))
        cond23 = (cond2 & (self.var.k_sat > 750))
        self.var.aCR[cond21] = (-0.4986 + (9 * (10 ** -5) * self.var.k_sat))[cond21]
        self.var.bCR[cond21] = (-2.1320 + (0.4778 * np.log(self.var.k_sat)))[cond21]
        self.var.aCR[cond22] = (-0.4986 + (9 * (10 ** -5) * 100))
        self.var.bCR[cond22] = (-2.1320 + (0.4778 * np.log(100)))
        self.var.aCR[cond23] = (-0.4986 + (9 * (10 ** -5) * 750))
        self.var.bCR[cond23] = (-2.1320 + (0.4778 * np.log(750)))

        # "Sandy clayey soil class"
        cond3 = (self.var.th_wilt >= 0.16) & (self.var.th_wilt <= 0.34) & (self.var.th_fc >= 0.25) & (self.var.th_fc <= 0.45) & (self.var.th_sat >= 0.40) & (self.var.th_sat <= 0.53)
        cond31 = (cond3 & (self.var.k_sat >= 5) & (self.var.k_sat <= 150))
        cond32 = (cond3 & (self.var.k_sat < 5))
        cond33 = (cond3 & (self.var.k_sat > 150))
        self.var.aCR[cond31] = (-0.5677 - (4 * (10 ** -5) * self.var.k_sat))[cond31]
        self.var.bCR[cond31] = (-3.7189 + (0.5922 * np.log(self.var.k_sat)))[cond31]
        self.var.aCR[cond32] = (-0.5677 - (4 * (10 ** -5) * 5))
        self.var.bCR[cond32] = (-3.7189 + (0.5922 * np.log(5)))
        self.var.aCR[cond33] = (-0.5677 - (4 * (10 ** -5) * 150))
        self.var.bCR[cond33] = (-3.7189 + (0.5922 * np.log(150)))

        # "Silty clayey soil class"
        cond4 = (self.var.th_wilt >= 0.20) & (self.var.th_wilt <= 0.42) & (self.var.th_fc >= 0.40) & (self.var.th_fc <= 0.58) & (self.var.th_sat >= 0.49) & (self.var.th_sat <= 0.58)
        cond41 = (cond4 & (self.var.k_sat >= 1) & (self.var.k_sat <= 150))
        cond42 = (cond4 & (self.var.k_sat < 1))
        cond43 = (cond4 & (self.var.k_sat > 150))
        self.var.aCR[cond41] = (-0.6366 + (8 * (10 ** -4) * self.var.k_sat))[cond41]
        self.var.bCR[cond41] = (-1.9165 + (0.7063 * np.log(self.var.k_sat)))[cond41]
        self.var.aCR[cond42] = (-0.6366 + (8 * (10 ** -4) * 1))
        self.var.bCR[cond42] = (-1.9165 + (0.7063 * np.log(1)))
        self.var.aCR[cond43] = (-0.6366 + (8 * (10 ** -4) * 150))
        self.var.bCR[cond43] = (-1.9165 + (0.7063 * np.log(150)))

        # Expand to soil compartments
        self.var.aCR_comp = self.var.aCR[...,self.var.layerIndex,:]
        self.var.bCR_comp = self.var.bCR[...,self.var.layerIndex,:]
        
    def dynamic(self):
        pass

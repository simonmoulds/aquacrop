#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import numpy as np
import netCDF4 as nc

# from hm import file_handling
from hm.api import open_hmdataarray

class SoilProfile(object):
    def __init__(self, model):
        self.model = model
        self.model.dz = self.model.config.SOIL_PROFILE['dzComp']  # TODO: rename to dz_comp
        self.model.dz_sum = np.cumsum(self.model.dz)              # TODO: consider name
        self.model.dz_layer = self.model.config.SOIL_PROFILE['dzLayer']
        self.model.nComp = len(self.model.dz)
        self.model.nLayer = len(self.model.dz_layer)
    def initial(self):
        pass
    def dynamic(self):
        pass

class SoilHydraulicParameters(object):
    def __init__(self, model):#, config_section_name):
        self.model = model
        SoilProfile(model)  # TODO: put this in the main model routine

    def initial(self):
        self.read_soil_hydraulic_parameters()
        self.compute_additional_soil_hydraulic_parameters()
        self.compute_capillary_rise_parameters()

    def read_soil_hydraulic_parameters(self):        
        soil_hydraulic_parameters = {
            'k_sat'  : self.model.config.SOIL_HYDRAULIC_PARAMETERS['saturatedHydraulicConductivityVarName'],
            'th_sat' : self.model.config.SOIL_HYDRAULIC_PARAMETERS['saturatedVolumetricWaterContentVarName'],
            'th_fc'  : self.model.config.SOIL_HYDRAULIC_PARAMETERS['fieldCapacityVolumetricWaterContentVarName'],
            'th_wilt': self.model.config.SOIL_HYDRAULIC_PARAMETERS['wiltingPointVolumetricWaterContentVarName']
        }
        for param, var_name in soil_hydraulic_parameters.items():
            # TODO: how to cope with broadcasting?
            # more general point: do we need to broadcast at all, now that we're not using numpy in the same way?
            arr = open_hmdataarray(
                self.model.config.SOIL_HYDRAULIC_PARAMETERS['soilHydraulicParametersNC'],
                var_name,
                self.model.domain
            )
            vars(self.model)[param] = np.broadcast_to(
                arr.values[None,None,:,:],
                (self.model.nFarm, self.model.nCrop, self.model.nLayer, self.model.domain.nxy)
            )            
            # d = file_handling.netcdf_to_arrayWithoutTime(
            #     self.model._configuration.SOIL_HYDRAULIC_PARAMETERS['soilHydraulicParametersNC'],
            #     var_name,
            #     cloneMapFileName=self.model.cloneMapFileName
            # )
            # d = d[self.model.landmask_layer].reshape(self.model.nLayer,self.model.domain.nxy)
            # vars(self.model)[param] = np.broadcast_to(
            #     d[None,None,:,:],
            #     (self.model.nFarm, self.model.nCrop, self.model.nLayer, self.model.domain.nxy)
            # )
    # def read_soil_hydraulic_parameters(self):
    #     pass
    
    def compute_additional_soil_hydraulic_parameters(self):
        zBot = np.cumsum(self.model.dz)
        zTop = zBot - self.model.dz
        zMid = (zTop + zBot) / 2
        dz_layer_bot = np.cumsum(self.model.dz_layer)
        dz_layer_top = dz_layer_bot - self.model.dz_layer        
        self.model.layerIndex = np.sum(((zMid[:,None] * np.ones((self.model.nLayer))[None,:]) > dz_layer_top), axis=1) - 1
        
        # The following is adapted from AOS_ComputeVariables.m, lines 129-139
        # "Calculate drainage characteristic (tau)
        self.model.tau = 0.0866 * (self.model.k_sat ** 0.35)
        self.model.tau = np.round(self.model.tau * 100) / 100
        self.model.tau[self.model.tau > 1] = 1
        self.model.tau[self.model.tau < 0] = 0

        # The following is adapted from AOS_ComputeVariables.m, lines 25
        self.model.th_dry = self.model.th_wilt / 2

        # transform certain soil properties to (ncrop, ncomp, ncell)
        soil_params = ['th_sat','th_fc','th_wilt','th_dry','k_sat','tau']
        for nm in soil_params:
            newnm = nm + '_comp'
            vars(self.model)[newnm] = vars(self.model)[nm][...,self.model.layerIndex,:]
            
    def compute_capillary_rise_parameters(self):
        # Function adapted from AOS_ComputeVariables.m, lines 60-127
        
        self.model.aCR = np.zeros((self.model.nFarm, self.model.nCrop, self.model.nLayer, self.model.domain.nxy))
        self.model.bCR = np.zeros((self.model.nFarm, self.model.nCrop, self.model.nLayer, self.model.domain.nxy))

        # "Sandy soil class"
        cond1 = (self.model.th_wilt >= 0.04) & (self.model.th_wilt <= 0.15) & (self.model.th_fc >= 0.09) & (self.model.th_fc <= 0.28) & (self.model.th_sat >= 0.32) & (self.model.th_sat <= 0.51)
        cond11 = (cond1 & (self.model.k_sat >= 200) & (self.model.k_sat <= 2000))
        cond12 = (cond1 & (self.model.k_sat < 200))
        cond13 = (cond1 & (self.model.k_sat > 2000))
        self.model.aCR[cond11] = (-0.3112 - (self.model.k_sat * (10 ** -5)))[cond11]
        self.model.bCR[cond11] = (-1.4936 + (0.2416 * np.log(self.model.k_sat)))[cond11]
        self.model.aCR[cond12] = (-0.3112 - (200 * (10 ** -5)))
        self.model.bCR[cond12] = (-1.4936 + (0.2416 * np.log(200)))
        self.model.aCR[cond13] = (-0.3112 - (2000 * (10 ** -5)))
        self.model.bCR[cond13] = (-1.4936 + (0.2416 * np.log(2000)))

        # "Loamy soil class"
        cond2 = (self.model.th_wilt >= 0.06) & (self.model.th_wilt <= 0.20) & (self.model.th_fc >= 0.23) & (self.model.th_fc <= 0.42) & (self.model.th_sat >= 0.42) & (self.model.th_sat <= 0.55)
        cond21 = (cond2 & (self.model.k_sat >= 100) & (self.model.k_sat <= 750))
        cond22 = (cond2 & (self.model.k_sat < 100))
        cond23 = (cond2 & (self.model.k_sat > 750))
        self.model.aCR[cond21] = (-0.4986 + (9 * (10 ** -5) * self.model.k_sat))[cond21]
        self.model.bCR[cond21] = (-2.1320 + (0.4778 * np.log(self.model.k_sat)))[cond21]
        self.model.aCR[cond22] = (-0.4986 + (9 * (10 ** -5) * 100))
        self.model.bCR[cond22] = (-2.1320 + (0.4778 * np.log(100)))
        self.model.aCR[cond23] = (-0.4986 + (9 * (10 ** -5) * 750))
        self.model.bCR[cond23] = (-2.1320 + (0.4778 * np.log(750)))

        # "Sandy clayey soil class"
        cond3 = (self.model.th_wilt >= 0.16) & (self.model.th_wilt <= 0.34) & (self.model.th_fc >= 0.25) & (self.model.th_fc <= 0.45) & (self.model.th_sat >= 0.40) & (self.model.th_sat <= 0.53)
        cond31 = (cond3 & (self.model.k_sat >= 5) & (self.model.k_sat <= 150))
        cond32 = (cond3 & (self.model.k_sat < 5))
        cond33 = (cond3 & (self.model.k_sat > 150))
        self.model.aCR[cond31] = (-0.5677 - (4 * (10 ** -5) * self.model.k_sat))[cond31]
        self.model.bCR[cond31] = (-3.7189 + (0.5922 * np.log(self.model.k_sat)))[cond31]
        self.model.aCR[cond32] = (-0.5677 - (4 * (10 ** -5) * 5))
        self.model.bCR[cond32] = (-3.7189 + (0.5922 * np.log(5)))
        self.model.aCR[cond33] = (-0.5677 - (4 * (10 ** -5) * 150))
        self.model.bCR[cond33] = (-3.7189 + (0.5922 * np.log(150)))

        # "Silty clayey soil class"
        cond4 = (self.model.th_wilt >= 0.20) & (self.model.th_wilt <= 0.42) & (self.model.th_fc >= 0.40) & (self.model.th_fc <= 0.58) & (self.model.th_sat >= 0.49) & (self.model.th_sat <= 0.58)
        cond41 = (cond4 & (self.model.k_sat >= 1) & (self.model.k_sat <= 150))
        cond42 = (cond4 & (self.model.k_sat < 1))
        cond43 = (cond4 & (self.model.k_sat > 150))
        self.model.aCR[cond41] = (-0.6366 + (8 * (10 ** -4) * self.model.k_sat))[cond41]
        self.model.bCR[cond41] = (-1.9165 + (0.7063 * np.log(self.model.k_sat)))[cond41]
        self.model.aCR[cond42] = (-0.6366 + (8 * (10 ** -4) * 1))
        self.model.bCR[cond42] = (-1.9165 + (0.7063 * np.log(1)))
        self.model.aCR[cond43] = (-0.6366 + (8 * (10 ** -4) * 150))
        self.model.bCR[cond43] = (-1.9165 + (0.7063 * np.log(150)))

        # Expand to soil compartments
        self.model.aCR_comp = self.model.aCR[...,self.model.layerIndex,:]
        self.model.bCR_comp = self.model.bCR[...,self.model.layerIndex,:]
    
    def dynamic(self):
        pass










            
# NOT USED:

# class SoilHydraulicParametersPoint(SoilHydraulicParameters):
#     def __init__(self, SoilHydraulicParametersPoint_variable, config_section_name):
#         super(AquaCrop, self).__init__(
#             SoilHydraulicParametersPoint_variable,
#             config_section_name
#         )
#         SoilProfilePoint(SoilHydraulicParameters_variable)
#         self.soil_hydrology = read_csv(
#             str(self.model.config.SOIL_HYDRAULIC_PARAMETERS['soilHydrologyFile']),
#             delimiter='\s+|\t',
#             header=None,
#             names=['layer','thickness','th_s','th_fc','th_wp','ksat'],
#             skiprows=0,
#             comment="%",
#             engine='python'
#         )
        
#     def read_soil_hydraulic_parameters(self):
#         soil_hydraulic_parameters = {
#             'th_s' : self.soil_hydrology['th_s'].values,
#             'th_fc' : self.soil_hydrology['th_fc'].values,
#             'th_wp' : self.soil_hydrology['th_wp'].values,
#             'k_sat' : self.soil_hydrology['ksat'].values
#         }        
#         for value, var_name in soil_hydraulic_parameters.items():
#             d = np.broadcast_to(value[:,None], self.model.nLayer, self.model.domain.nxy)
#             vars(self.model)[param] = np.broadcast_to(
#                 d[None,None,:,:],
#                 (self.model.nFarm, self.model.nCrop, self.model.nLayer, self.model.domain.nxy)
#             )

# class SoilHydraulicParametersGrid(SoilHydraulicParameters):
#     def __init__(self, SoilHydraulicParametersGrid_variable, config_section_name):
#         self.model = SoilHydraulicParametersGrid_variable
#         SoilProfileGrid(SoilHydraulicParametersGrid_variable)

#     def read_soil_hydraulic_parameters(self):        
#         soil_hydraulic_parameters = {
#             'k_sat'  : self.model.config.SOIL_HYDRAULIC_PARAMETERS['saturatedHydraulicConductivityVarName'],
#             'th_sat' : self.model.config.SOIL_HYDRAULIC_PARAMETERS['saturatedVolumetricWaterContentVarName'],
#             'th_fc'  : self.model.config.SOIL_HYDRAULIC_PARAMETERS['fieldCapacityVolumetricWaterContentVarName'],
#             'th_wilt': self.model.config.SOIL_HYDRAULIC_PARAMETERS['wiltingPointVolumetricWaterContentVarName']
#         }
#         for param, var_name in soil_hydraulic_parameters.items():
#             # TODO: check dimensions are as expected
#             d = file_handling.netcdf_to_arrayWithoutTime(
#                 self.model._configuration.SOIL_HYDRAULIC_PARAMETERS['soilHydraulicParametersNC'],
#                 var_name,
#                 cloneMapFileName=self.model.cloneMapFileName
#             )
#             d = d[self.model.landmask_layer].reshape(self.model.nLayer,self.model.domain.nxy)
#             vars(self.model)[param] = np.broadcast_to(
#                 d[None,None,:,:],
#                 (self.model.nFarm, self.model.nCrop, self.model.nLayer, self.model.domain.nxy)
#             )

# class SoilProfilePoint(object):
#     def __init__(self, SoilProfile_variable):
#         self.model = SoilProfile_variable
#         soil_profile = read_csv(
#             self.model.config.SOIL_PROFILE['soilProfileFile'],
#             delimiter='\s+|\t',
#             header=None,
#             names=['compartment','thickness','layer'],
#             skiprows=0,
#             comment="%",
#             engine='python'
#         )
#         soil_hydrology = read_csv(
#             str(self.model.config.SOIL_HYDRAULIC_PARAMETERS['soilHydrologyFile']),
#             delimiter='\s+|\t',
#             header=None,
#             names=['layer','thickness','th_s','th_fc','th_wp','ksat'],
#             skiprows=0,
#             comment="%",
#             engine='python')
        
#         self.model.dz = np.array(soil_profile['thickness'])
#         self.model.dz_sum = np.cumsum(self.model.dz)
#         self.model.dz_layer = np.array(soil_hydrology['thickness'].values)
#         self.model.nComp = len(self.model.dz)
#         self.model.nLayer = len(self.model.dz_layer)

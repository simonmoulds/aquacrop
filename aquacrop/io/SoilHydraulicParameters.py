#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import numpy as np

from hm.api import open_hmdataarray

import aquacrop_fc


class SoilProfile(object):
    def __init__(self, model):
        self.model = model
        # TODO: rename to dz_comp
        self.model.dz = np.float64(
            self.model.config.SOIL_PROFILE['dzComp']
        )
        self.model.dz_layer = np.float64(
            self.model.config.SOIL_PROFILE['dzLayer']
        )
        self.model.dz_sum = np.cumsum(self.model.dz)
        self.model.nComp = len(self.model.dz)
        self.model.nLayer = len(self.model.dz_layer)

    def initial(self):
        pass

    def dynamic(self):
        pass


class SoilHydraulicParameters(object):
    def __init__(self, model):
        self.model = model
        self.config = self.model.config.SOIL_HYDRAULIC_PARAMETERS
        SoilProfile(model)  # TODO: put this in the main model routine

    def initial(self):
        self.read_soil_hydraulic_parameters()
        self.compute_additional_soil_hydraulic_parameters()
        
    def read_soil_hydraulic_parameters(self):

        # TODO: optionally read directly from config file
        
        soil_hydraulic_parameters = {
            'k_sat'  : self.config['saturatedHydraulicConductivityVarName'],
            'th_sat' : self.config['saturatedVolumetricWaterContentVarName'],
            'th_fc'  : self.config['fieldCapacityVolumetricWaterContentVarName'],
            'th_wilt': self.config['wiltingPointVolumetricWaterContentVarName']
        }
        for param, var_name in soil_hydraulic_parameters.items():

            if param in self.config.keys():
                # 1 - Try to read from config file
                parameter_values = np.array(self.config[param]) 
                if (len(parameter_values) == self.model.nLayer):
                    vars(self.model)[param] = np.require(
                        np.broadcast_to(
                            parameter_values,
                            (self.model.nLayer,
                             self.model.domain.nxy)
                        ),
                        requirements=['A','O','W','F']
                    )
                else:
                    raise ValueError(
                        "Error reading parameter " + param
                        + " from configuration file: length"
                        + " of parameter list must equal number"
                        + " of farms in simulation"
                    )
                
            else:        
                # 2 - Try to read from netCDF file
                try:
                    arr = open_hmdataarray(
                        self.config['soilHydraulicParametersNC'],
                        var_name,
                        self.model.domain,
                        self.model.config.SOIL_HYDRAULIC_PARAMETERS['is_1d'],
                        self.model.config.SOIL_HYDRAULIC_PARAMETERS['xy_dimname'],
                    )
                    vars(self.model)[param] = np.require(
                        arr.values,
                        requirements=['A','O','W','F']
                    )
                except:
                    raise ValueError()  # TODO
                    
    def compute_additional_soil_hydraulic_parameters(self):
        zBot = np.cumsum(self.model.dz)
        zTop = zBot - self.model.dz
        zMid = (zTop + zBot) / 2
        dz_layer_bot = np.cumsum(self.model.dz_layer)
        dz_layer_top = dz_layer_bot - self.model.dz_layer
        self.model.layerIndex = np.sum(
            ((zMid[:, None] * np.ones((self.model.nLayer))[None, :]) > dz_layer_top),
            axis=1
        ) - 1

        flt_params_to_compute = ['aCR','bCR','tau','th_dry']
        for param in flt_params_to_compute:
            vars(self.model)[param] = np.require(
                np.zeros(
                    (self.model.nLayer, self.model.domain.nxy),
                    dtype=np.float64,
                ),
                requirements=['A','O','W','F']
            )
            
        aquacrop_fc.soil_hydraulic_parameters_w.compute_soil_h_parameters_w(
            np.asfortranarray(self.model.aCR.T),  # TODO
            np.asfortranarray(self.model.bCR.T),  # TODO
            np.asfortranarray(self.model.tau.T),  # TODO
            np.asfortranarray(self.model.th_dry.T),  # TODO
            self.model.th_fc.T,
            self.model.th_sat.T,
            self.model.th_wilt.T,
            self.model.k_sat.T,
            int(1),
            self.model.nLayer,
            self.model.domain.nxy
        )
        
    def dynamic(self):
        pass



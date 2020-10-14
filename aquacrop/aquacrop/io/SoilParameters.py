#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import sqlite3
from importlib_resources import path

from hm.api import open_hmdataarray
from .utils import read_parameter_from_sqlite
from . import data

import aquacrop_fc


class SoilParameters(object):
    def __init__(self, model):
        self.model = model
        # NOT CURRENTLY IMPLEMENTED:
        # self.model.adjustReadilyAvailableWater = \
        #     self.model.config.SOIL_PARAMETERS['adjustReadilyAvailableWater']
        self.model.adjustCurveNumber = self.model.config.SOIL_PARAMETERS['adjustCurveNumber']
        self.load_soil_parameter_database()

    def initial(self):
        self.read_soil_parameters()

        self.model.CNbot = np.zeros((self.model.domain.nxy))
        self.model.CNtop = np.zeros((self.model.domain.nxy))
        self.model.REW = np.zeros((self.model.domain.nxy))
        
        aquacrop_fc.soil_parameters_w.compute_soil_parameters_w(
            self.model.CNbot.T,
            self.model.CNtop.T,
            self.model.REW.T,
            self.model.zGerm.T,
            self.model.CN.T,
            self.model.th_fc.T,
            self.model.th_dry.T,
            self.model.EvapZsurf.T,
            self.model.zCN.T,
            self.model.dz_sum,
            self.model.nComp, self.model.nLayer, self.model.domain.nxy
        )        
        
    def load_soil_parameter_database(self):
        with path(data, 'soil_parameter_database.sqlite3') as db_path:
            try:
                db_path = db_path.resolve()
            except FileNotFoundError:
                pass
            self.model.SoilParameterDatabase = sqlite3.connect(str(db_path))

    def read_soil_parameters(self):
        soil_parameters = [
            'EvapZsurf', 'EvapZmin', 'EvapZmax', 'Kex',
            'fevap', 'fWrelExp', 'fwcc',
            'CN', 'zCN', 'zGerm', 'zRes', 'fshape_cr'
        ]
        for param in soil_parameters:
            try:
                arr = open_hmdataarray(
                    self.model.config.SOIL_PARAMETERS['filename'],
                    param,
                    self.model.domain,
                    self.model.config.SOIL_PARAMETERS['is_1d'],
                    self.model.config.SOIL_PARAMETERS['xy_dimname'],
                )
                vars(self.model)[param] = np.require(
                    arr.values,
                    requirements=['A','O','W','F']
                )
                
            except:
                try:
                    parameter_value = read_parameter_from_sqlite(
                        self.model.SoilParameterDatabase,
                        param
                    )
                    parameter_value = np.array(
                        parameter_value[0],
                        dtype=np.float64
                    )
                    vars(self.model)[param] = np.require(
                        np.full(self.model.domain.nxy, parameter_value),
                        requirements=['A','O','W','F']
                    )
                except:
                    pass
            
    def dynamic(self):
        pass

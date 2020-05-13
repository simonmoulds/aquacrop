#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import sqlite3
from importlib_resources import path

# from .AquaCropConfiguration import interpret_logical_string
from hm.api import open_hmdataarray
from .utils import read_parameter_from_sqlite
from . import data


class SoilParameters(object):
    def __init__(self, model):  # , config_section_name):
        self.model = model
        self.model.adjustReadilyAvailableWater = self.model.config.SOIL_PARAMETERS[
            'adjustReadilyAvailableWater']
        self.model.adjustCurveNumber = self.model.config.SOIL_PARAMETERS['adjustCurveNumber']
        self.load_soil_parameter_database()

    def initial(self):
        self.read_soil_parameters()
        if not self.model.adjustReadilyAvailableWater:
            self.compute_readily_available_water()
        self.compute_curve_number_limits()
        self.compute_weighting_factor_for_cn_adjustment()
        self.adjust_zgerm()

    def load_soil_parameter_database(self):
        with path(data, 'soil_parameter_database.sqlite3') as db_path:
            try:
                db_path = db_path.resolve()
            except FileNotFoundError:
                pass
            self.model.SoilParameterDatabase = sqlite3.connect(str(db_path))

    def compute_curve_number_limits(self):
        # The following is adapted from AOS_ComputeVariables.m, lines 147-151
        # "Calculate upper and lower curve numbers"

        # N.B these seem to be adapted from Eq 3.7g and Eq 3.7h,
        # but the values of the coefficients do not match.
        self.model.CNbot = np.round(1.4 * (np.exp(-14 * np.log(10))) + (0.507 * self.model.CN) - (
            0.00374 * self.model.CN ** 2) + (0.0000867 * self.model.CN ** 3))
        self.model.CNtop = np.round(5.6 * (np.exp(-14 * np.log(10))) + (2.33 * self.model.CN) - (
            0.0209 * self.model.CN ** 2) + (0.000076 * self.model.CN ** 3))

    def compute_readily_available_water(self):
        self.model.REW = (np.round(
            (1000 * (self.model.th_fc[..., 0, :] - self.model.th_dry[..., 0, :]) * self.model.EvapZsurf)))

    def compute_weighting_factor_for_cn_adjustment(self):
        # N.B. in the case no compartment is shallower than the specified zCN,
        # wrel equals 1 for the top compartment and zero for all others.
        zcn = np.broadcast_to(self.model.zCN[..., None, :], (
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.domain.nxy))
        dz_sum = np.broadcast_to(
            self.model.dz_sum[None, None, :, None],
            (self.model.nFarm, self.model.nCrop,
             self.model.nComp, self.model.domain.nxy)
        ).copy()
        dz_sum[dz_sum > zcn] = zcn[dz_sum > zcn]
        wx = (1.016 * (1 - np.exp(-4.16 * (dz_sum / zcn))))
        xx = np.concatenate(
            (np.zeros((self.model.nFarm, self.model.nCrop, 1, self.model.domain.nxy)),
             wx[:, :, :-1, :]),
            axis=2
        )
        self.model.weighting_factor_for_cn_adjustment = np.clip(
            (wx - xx), 0, 1)

    def adjust_zgerm(self):
        # Force zGerm to have a maximum value equal to the depth of the
        # deepest soil compartment
        zgerm = np.copy(self.model.zGerm)
        zgerm[zgerm > np.sum(self.model.dz, axis=0)] = np.sum(
            self.model.dz, axis=0)
        self.model.zGerm = zgerm.copy()

    def dynamic(self):
        pass

    def read_soil_parameters(self):
        soil_parameters = [
            'EvapZsurf', 'EvapZmin', 'EvapZmax', 'Kex',
            'fevap', 'fWrelExp', 'fwcc',
            'CN', 'zCN', 'zGerm', 'zRes', 'fshape_cr'
        ]
        for param in soil_parameters:
            try:
                arr = open_hmdataarray(
                    self.model.config.SOIL_PARAMETERS['soilParametersNC'],
                    param,
                    self.model.domain
                )
                vars(self.model)[param] = np.broadcast_to(
                    arr.values,
                    (self.model.nFarm, self.model.nCrop, self.model.domain.nxy)
                )
                print(self.model.config.SOIL_PARAMETERS['soilParametersNC'])
                print(arr.values)
                print(param)
            except TypeError:
                try:
                    parameter_value = read_parameter_from_sqlite(
                        self.model.SoilParameterDatabase,
                        param
                    )
                    parameter_value = np.array(
                        parameter_value[0], dtype=np.float64)
                    vars(self.model)[param] = np.full(
                        (self.model.nFarm, self.model.nCrop, self.model.domain.nxy),
                        parameter_value
                    )
                except:
                    pass

# def read_params(fn):
#     with open(fn) as f:
#         content = f.read().splitlines()
#     # remove commented lines
#     content = [x for x in content if re.search('^(?!%).*', x)]
#     content = [re.split('\s*:\s*', x) for x in content]
#     params = {}
#     for x in content:
#         if len(x) > 1:
#             nm = x[0]
#             val = x[1]
#             params[nm] = val
#     return params

# class SoilParametersPoint(SoilParameters):

#     def read_soil_parameters(self):
#         soil_parameters = [
#             'EvapZsurf','EvapZmin', 'EvapZmax', 'Kex',
#             'fevap', 'fWrelExp', 'fwcc',
#             'CN', 'zCN', 'zGerm', 'zRes', 'fshape_cr'
#         ]
#         soil_parameter_values = read_params(self.model._configuration.SOIL_PARAMETERS['soilParameterFile'])

#         for param in soil_parameters:
#             read_from_file = (param in soil_parameter_values.keys())
#             if read_from_file:
#                 d = soil_parameter_values[param]
#                 d = np.broadcast_to(d[None,None,:], (self.model.nFarm,self.model.nCrop, self.model.domain.nxy))
#                 vars(self.model)[param] = d.copy()

#             else:
#                 try:
#                     parameter_value = file_handling.read_soil_parameter_from_sqlite(
#                         self.model.SoilParameterDatabase,
#                         param
#                     )
#                     parameter_value = np.array(parameter_value[0], dtype=np.float64)
#                     vars(self.model)[param] = np.full(
#                         (self.model.nFarm, self.model.nCrop, self.model.domain.nxy),
#                         parameter_value)
#                 except:
#                     pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .constants import eps


class SaturationVapourPressure(object):
    def __init__(self, model):
        self.model = model

    def initial(self):
        self.model.es_min = np.zeros((self.model.domain.nxy))
        self.model.es_max = np.zeros((self.model.domain.nxy))
        self.model.es_mean = np.zeros((self.model.domain.nxy))

    def compute_saturation_vapour_pressure(self):
        def es(T): return 610.8*np.exp((17.27*(T-273.15))/((T-273.15)+237.3))
        if self.model.config.has_min_daily_temperature:
            self.model.es_min = es(self.model.tmin.values)
        if self.model.config.has_max_daily_temperature:
            self.model.es_max = es(self.model.tmax.values)
        if (self.model.config.has_min_daily_temperature
                and self.model.config.has_max_daily_temperature):
            self.model.es_mean = (self.model.es_min + self.model.es_max) / 2.
        else:
            self.model.es_mean = es(self.model.tmean)

    def dynamic(self):
        self.compute_saturation_vapour_pressure()


class ActualVapourPressure(object):
    def __init__(self, model):
        self.model = model

    def initial(self):
        self.model.ea_mean = np.zeros((self.model.domain.nxy))

    def compute_actual_vapour_pressure_from_specific_humidity(self):
        """Compute actual vapour pressure given specific
        humidity, using equations from Bolton (1980; 
        https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html)
        """
        def ea(Pres, Q, eps): return (Q * Pres) / ((1 - eps) * Q + eps)
        self.model.ea_mean = ea(
            self.model.surface_pressure,
            self.model.specific_humidity,
            eps
        )

    def compute_actual_vapour_pressure_from_relative_humidity_eq17(self):
        self.model.ea_mean = (
            (self.model.es_min * self.model.max_relative_humidity.values / 100.)
            + (self.model.es_max * self.model.min_relative_humidity.values / 100.)
        ) / 2

    def compute_actual_vapour_pressure_from_relative_humidity_eq18(self):
        self.model.ea_mean = self.model.es_min * self.model.max_relative_humidity.values / 100.

    def compute_actual_vapour_pressure_from_relative_humidity_eq19(self):
        self.model.ea_mean = self.model.es_mean * self.model.mean_relative_humidity.values / 100.

    def can_use_fao_equation_17(self):
        return (
            self.model.config.has_min_daily_temperature
            and self.model.config.has_max_daily_temperature
            and self.model.config.has_min_relative_humidity
            and self.model.config.has_max_relative_humidity
        )

    def can_use_fao_equation_18(self):
        return (
            self.model.config.has_min_daily_temperature
            and self.model.config.has_max_relative_humidity
        )

    def can_use_fao_equation_19(self):
        return (
            self.model.config.has_max_daily_temperature
            and self.model.config.has_min_daily_temperature
            and self.model.config.has_mean_relative_humidity
        )

    def dynamic(self):
        if self.can_use_fao_equation_17():
            self.compute_actual_vapour_pressure_from_relative_humidity_eq17()
        elif self.can_use_fao_equation_18():
            self.compute_actual_vapour_pressure_from_relative_humidity_eq18()
        elif self.can_use_fao_equation_19():
            self.compute_actual_vapour_pressure_from_relative_humidity_eq19()
        elif self.model.config.has_specific_humidity:
            self.compute_actual_vapour_pressure_from_specific_humidity()


class VapourPressureDeficit(object):
    def __init__(self, model):
        self.model = model

    def initial(self):
        self.model.vapour_pressure_deficit = np.zeros((self.model.domain.nxy))

    def compute_vapour_pressure_deficit(self):
        self.model.vapour_pressure_deficit = np.clip(
            self.model.es_mean - self.model.ea_mean,
            0,
            None
        )

    def dynamic(self):
        self.compute_vapour_pressure_deficit()


class VapourPressure(object):
    def __init__(self, VapourPressure_variable):
        self.saturation_vapour_pressure_module = \
            SaturationVapourPressure(VapourPressure_variable)
        self.actual_vapour_pressure_module = \
            ActualVapourPressure(VapourPressure_variable)
        self.vapour_pressure_deficit_module = \
            VapourPressureDeficit(VapourPressure_variable)

    def initial(self):
        self.saturation_vapour_pressure_module.initial()
        self.actual_vapour_pressure_module.initial()
        self.vapour_pressure_deficit_module.initial()

    def dynamic(self):
        self.saturation_vapour_pressure_module.dynamic()
        self.actual_vapour_pressure_module.dynamic()
        self.vapour_pressure_deficit_module.dynamic()

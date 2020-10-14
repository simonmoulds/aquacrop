#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from hm.model import Model
from hm.reporting import Reporting

from .extraterrestrialradiation import ExtraterrestrialRadiation
from .inputdata import (LongwaveRadiation,
                        ShortwaveRadiation,
                        RelativeHumidity,
                        SpecificHumidity,
                        SurfacePressure,
                        Temperature,
                        Wind)
from .netradiation import NetRadiation
from .vapourpressure import VapourPressure
from . import variable_list

import logging
logger = logging.getLogger(__name__)

class PenmanMonteith(Model):
    def __init__(self, config, time, domain, init=None):
        super(PenmanMonteith, self).__init__(
            config,
            time,
            domain,
            is_1d=True,
            init=init
        )
        self.temperature_module = Temperature(self)
        # self.longwave_radiation_module = LongwaveRadiation(self)
        self.shortwave_radiation_module = ShortwaveRadiation(self)
        self.extraterrestrial_radiation_module = ExtraterrestrialRadiation(
            self)
        self.relative_humidity_module = RelativeHumidity(self)
        self.specific_humidity_module = SpecificHumidity(self)
        self.surface_pressure_module = SurfacePressure(self)
        self.vapour_pressure_module = VapourPressure(self)
        self.net_radiation_module = NetRadiation(self)
        self.wind_module = Wind(self)
        self.ETref = np.zeros((self.domain.nxy))
        
    def initial(self):
        # self.longwave_radiation_module.initial()
        self.shortwave_radiation_module.initial()
        self.extraterrestrial_radiation_module.initial()
        self.relative_humidity_module.initial()
        self.specific_humidity_module.initial()
        self.surface_pressure_module.initial()
        self.vapour_pressure_module.initial()
        self.net_radiation_module.initial()
        self.temperature_module.initial()
        self.wind_module.initial()
        
        self.reporting_module = Reporting(self, variable_list)
        self.reporting_module.initial()

    def penman_monteith(self):
        """Function implementing the Penman-Monteith
        equation (*NOT* FAO version)
        """
        cp = 0.001013     # specific heat of air 1013 [MJ kg-1 K-1]
        TimeStepSecs = 86400        # timestep in seconds
        rs = 70           # surface resistance, 70 [s m-1]
        R = 287.058      # Universal gas constant [J kg-1 K-1]
        # ratio of water vapour/dry air molecular weights [-]
        eps = 0.622
        # g            = 9.81         # gravitational constant [m s-2]

        rho = self.surface_pressure.values / (self.tmean.values * R)  # density of air [kg m-3]

        # latent heat [MJ kg-1]
        latent_heat = (2.501 - (0.002361 * (self.tmean.values - 273.15)))
        
        # slope of vapour pressure [kPa K-1]
        deltop = 4098. * \
            (0.6108 * np.exp((17.27 * (self.tmean.values - 273.15)) /
                             ((self.tmean.values - 273.15) + 237.3)))
        delbase = ((self.tmean.values - 273.15) + 237.3) ** 2
        delta = deltop / delbase

        # psychrometric constant [kPa K-1]
        gamma = cp * (self.surface_pressure.values / 1000) / (eps * latent_heat)
        
        # aerodynamic resistance [m s-1]
        z = 10  # height of wind speed variable (10 meters above surface)
        Wsp_2 = self.wind.values * 4.87 / (np.log(67.8 * z - 5.42))
        ra = np.divide(208., Wsp_2, out=np.zeros_like(Wsp_2), where=Wsp_2 != 0)

        # Penman-Monteith equation (NB unit conversion)
        PETtop = np.maximum(
            ((delta * 1e3)
             * self.net_radiation
             + rho
             * (cp * 1e6)
             * np.divide(
                 self.vapour_pressure_deficit,
                 ra,
                 out=np.zeros_like(ra),
                 where=ra != 0)),
            1)

        PETbase = np.maximum(
            ((delta * 1e3)
             + (gamma * 1e3)
             * (1 + np.divide(rs, ra, out=np.zeros_like(ra), where=ra != 0))),
            1)
        PET = np.maximum(PETtop / PETbase, 0)

        PETmm = np.maximum((PET / (latent_heat * 1e6) * TimeStepSecs), 0)
        self.ETref = PETmm.copy()

        # FAO Penman-Monteith:
        # PETtop = (
        #     (0.408 * delta * self.net_radiation * 0.086400)
        #     + (gamma
        #        * (900 / self.tmean)
        #        * Wsp_2
        #        * self.vapour_pressure_deficit / 1e3)
        # )
        # PETbot = (delta + gamma * (1 + 0.34 * Wsp_2))
        # PETmm = PETtop / PETbot
        # self.ETref = PETmm.copy()

    def dynamic(self):
        # logger.info("Reading forcings for time %s", self._modelTime)
        self.temperature_module.dynamic()
        # self.longwave_radiation_module.dynamic()
        self.shortwave_radiation_module.dynamic()
        self.extraterrestrial_radiation_module.dynamic()
        self.relative_humidity_module.dynamic()
        self.specific_humidity_module.dynamic()
        self.surface_pressure_module.dynamic()
        self.vapour_pressure_module.dynamic()
        self.net_radiation_module.dynamic()
        self.wind_module.dynamic()
        self.penman_monteith()
        self.reporting_module.dynamic()

# Stefan-Boltzmann [MJ m-2 K-4]
SIGMA = 4.903e-9

# Specific heat of air [MJ kg-1 K-1]
CP = 0.001013

# Ratio molecular weight of water vapour/dry air
EPS = 0.622

# empirical constant in evaporative demand equation
FACTOR_CANOPY = 1.
FACTOR_SOIL = 0.75
FACTOR_WATER = 0.5

# surface albedo
ALBEDO_CANOPY = 0.23
ALBEDO_SOIL = 0.15
ALBEDO_WATER = 0.05


# class PenmanMonteith2(Model):
#     def __init__(self, configuration, modelTime, initialState):
#         super(PenmanMonteith, self).__init__(
#             configuration,
#             modelTime,
#             initialState)
#         self.temperature_module = Temperature(self)
#         # self.longwave_radiation_module = LongwaveRadiation(self)
#         self.shortwave_radiation_module = ShortwaveRadiation(self)
#         self.extraterrestrial_radiation_module = ExtraterrestrialRadiation(
#             self)
#         self.relative_humidity_module = RelativeHumidity(self)
#         self.specific_humidity_module = SpecificHumidity(self)
#         self.surface_pressure_module = SurfacePressure(self)
#         self.vapour_pressure_module = VapourPressure(self)
#         self.net_radiation_module = NetRadiation(self)
#         self.wind_module = Wind(self)

#     def initial(self):
#         # self.longwave_radiation_module.initial()
#         self.shortwave_radiation_module.initial()
#         self.extraterrestrial_radiation_module.initial()
#         self.relative_humidity_module.initial()
#         self.specific_humidity_module.initial()
#         self.surface_pressure_module.initial()
#         self.vapour_pressure_module.initial()
#         self.net_radiation_module.initial()
#         self.temperature_module.initial()
#         self.wind_module.initial()

#         # For extraterrestrial radiation
#         latitudes = np.broadcast_to(
#             self.model.domain.y[:, None],
#             (self.model.domain.ny, self.model.domain.nx)
#         )
#         self.latitudes = latitudes[self.model.domain.mask]

#     def saturated_vapour_pressure(self):
#         pass

#     def compute_actual_vapour_pressure(self):
#         if self.can_use_fao_equation_14():
#             self.compute_actual_vapour_pressure_from_dewpoint_temperature_eq14()
#         elif self.can_use_fao_equation_15():
#             self.compute_actual_vapour_pressure_from_psychrometric_data_eq15()
#         if self.can_use_fao_equation_17():
#             self.compute_actual_vapour_pressure_from_relative_humidity_eq17()
#         elif self.can_use_fao_equation_18():
#             self.compute_actual_vapour_pressure_from_relative_humidity_eq18()
#         elif self.can_use_fao_equation_19():
#             self.compute_actual_vapour_pressure_from_relative_humidity_eq19()
#         elif self.model.config.has_specific_humidity:
#             self.compute_actual_vapour_pressure_from_specific_humidity()

#     def can_use_fao_equation_14(self):
#         return self.config.has_dewpoint_temperature

#     def can_use_fao_equation_15(self):
#         return False

#     def can_use_fao_equation_17(self):
#         return (
#             self.model.config.has_min_daily_temperature
#             and self.model.config.has_max_daily_temperature
#             and self.model.config.has_min_relative_humidity
#             and self.model.config.has_max_relative_humidity
#         )

#     def can_use_fao_equation_18(self):
#         return (
#             self.model.config.has_min_daily_temperature
#             and self.model.config.has_max_relative_humidity
#         )

#     def can_use_fao_equation_19(self):
#         return (
#             self.model.config.has_max_daily_temperature
#             and self.model.config.has_min_daily_temperature
#             and self.model.config.has_mean_relative_humidity
#         )

#     def compute_actual_vapour_pressure_from_dewpoint_temperature_eq14(self):
#         """Compute actual vapour pressure using FAO56 Eq. 14.

#         Original equation gives value in kPa; here we convert to mbar
#         """
#         self.ea_mean = (
#             0.610588
#             * np.exp(
#                 (17.32491 * self.tdew)
#                 / (self.tdew + 238.102)
#             )
#         )
#         # convert from kPa to mbar
#         self.ea_mean *= 10.

#     def compute_actual_vapour_pressure_from_psychrometric_data_eq15(self):
#         pass

#     def compute_actual_vapour_pressure_from_relative_humidity_eq17(self):
#         self.ea_mean = (
#             (self.es_min * self.max_relative_humidity / 100.)
#             + (self.es_max * self.min_relative_humidity / 100.)
#         ) / 2
#         # convert from kPa to mbar
#         self.ea_mean *= 10.

#     def compute_actual_vapour_pressure_from_relative_humidity_eq18(self):
#         self.model.ea_mean = (
#             self.model.es_min
#             * self.model.max_relative_humidity
#             / 100.
#         )
#         # convert from kPa to mbar
#         self.ea_mean *= 10.

#     def compute_actual_vapour_pressure_from_relative_humidity_eq19(self):
#         self.model.ea_mean = (
#             self.model.es_mean
#             * self.model.mean_relative_humidity
#             / 100.
#         )
#         # convert from kPa to mbar
#         self.ea_mean *= 10.

#     def actual_vapour_pressure_from_specific_humidity(self):
#         """Compute actual vapour pressure given specific
#         humidity, using equations from Bolton (1980;
#         https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html)
#         """
#         def ea(Pres, Q, eps): return (Q * Pres) / ((1 - eps) * Q + eps)
#         self.ea_mean = ea(
#             self.surface_pressure.values, self.specific_humidity.values, eps
#         )
#         # convert from kPa to mbar
#         self.ea_mean *= 10.

#     def compute_saturated_vapour_pressure(self):
#         # def es(T): return 610.8*np.exp((17.27*(T-273.15))/((T-273.15)+237.3))
#         # TODO: Kelvin/degC ?
#         # es_mean is in mbar
#         def es(T): return 0.610588 * np.exp((17.32491 * T) / (T + 238.102))
#         if self.config.has_min_daily_temperature:
#             self.es_min = es(self.tmin.values) * 10.
#         if self.config.has_max_daily_temperature:
#             self.es_max = es(self.tmax.values) * 10.
#         if (self.config.has_min_daily_temperature
#                 and self.config.has_max_daily_temperature):
#             self.es_mean = (self.es_min + self.es_max) / 2.
#         else:
#             self.es_mean = es(self.tmean) * 10.

#     def compute_extraterrestrial_radiation(self):
#         """Compute extraterrestrial radiation (MJ m-2 d-1)"""
#         LatRad = self.latitudes * np.pi / 180.0
#         declin = (
#             0.4093 * (np.sin(((2.0 * np.pi * self.time.doy) / 365.0) - 1.405)))
#         arccosInput = (-(np.tan(LatRad)) * (np.tan(declin)))
#         arccosInput = np.clip(arccosInput, -1, 1)
#         sunangle = np.arccos(arccosInput)
#         distsun = 1 + 0.033 * \
#             (np.cos((2 * np.pi * self.time.doy) / 365.0))
#         self.extraterrestrial_radiation = (
#             ((24 * 60 * 0.082) / np.pi)
#             * distsun
#             * (sunangle
#                * (np.sin(LatRad))
#                * (np.sin(declin))
#                + (np.cos(LatRad))
#                * (np.cos(declin))
#                * (np.sin(sunangle)))
#         )

#     def compute_net_longwave_radiation(self):
#         """Compute net longwave radiation (MJ m-2 d-1)."""

#         # dimensionless (actual vapour pressure in mbar)
#         net_emissivity = (0.56 - 0.079 * np.sqrt(self.ea_mean))

#         # MJ m-2 d-1
#         clear_sky_solar_radiation = (
#             self.extraterrestrial_radiation
#             * (0.75 + (2. * 10 ** -5 * self.elevation))
#         )

#         # dimensionless (radiation variables in MJ m-2 d-1)
#         cloud_cover_factor = (
#             1.8 *
#             np.divide(
#                 self.shortwave_radiation,
#                 clear_sky_solar_radiation,
#                 out=np.zeros_like(self.shortwave_radiation),
#                 where=clear_sky_solar_radiation != 0
#             )
#             - 0.35
#         )
#         cloud_cover_factor[cloud_cover_factor < 0.] = 0.05
#         cloud_cover_factor[cloud_cover_factor > 1.] = 1.

#         # Net longwave radiation in MJ m-2 d-1
#         # Stefan Boltzmann coefficient in MJ m-2 K-4
#         # Tmean in Kelvin
#         self.net_longwave_radiation = (
#             SIGMA
#             * self.tmean ** 4
#             * net_emissivity
#             * cloud_cover_factor
#         )

#     def compute_latent_heat_of_vaporization(self):
#         """Compute latent heat of vaporization (MJ kg-1)."""
#         # Tmean in K
#         self.latent_heat_of_vaporization = (
#             2501. - 2.375 * (self.tmean - 273.15)
#         ) / 1000.

#     def compute_psychrometric_constant(self):
#         """Compute psychrometric constant (mbar K-1)."""
#         # TODO: make sure surface_pressure is in kPa
#         psychrometric_constant_sea_level = (
#             cp
#             * (self.surface_pressure * 10)
#             / (eps * self.latent_heat_of_vaporization)
#         )
#         # correction for altitude (elevation in masl)
#         self.psychrometric_constant = (
#             psychrometric_constant_sea_level
#             * ((293. - 0.0065 * self.elevation) / 293.) ** 5.26
#         )

#     def compute_slope_of_vapour_pressure_curve(self):
#         """Compute slope of vapour pressure curve (mbar K-1)."""
#         self.slope_of_vapour_pressure_curve = (
#             (238.102 * 17.32491 * self.es_mean)
#             / (((self.tmean - 273.15) + 238.102) ** 2)
#         )

#     def compute_wind_coefficient(self):
#         """Compute wind coefficient (-)."""
#         delta_T = np.max(self.tmax - self.tmin, 0.)
#         self.wind_coefficient = np.maximum(
#             0.54 + 0.35 * ((delta_T - 12.) / 4.),
#             0.
#         )

#     def compute_net_absorbed_radiation(self, albedo):
#         """Compute net absorbed radiation (MJ m-2 d-1)."""
#         self.compute_net_longwave_radiation()

#         def rna(albedo):
#             rna = (
#                 (1. - albedo)
#                 * self.shortwave_radiation
#                 - self.net_longwave_radiation
#             ) / (1e6 * self.latent_heat_of_vaporization)
#             return rna

#         self.net_absorbed_radiation_canopy = rna(ALBEDO_CANOPY)
#         self.net_absorbed_radiation_soil = rna(ALBEDO_SOIL)
#         self.net_absorbed_radiation_water = rna(ALBEDO_WATER)

#     def compute_vapour_pressure_deficit(self):
#         """Compute vapour pressure deficit (mbar)."""
#         self.vapour_pressure_deficit = (
#             self.saturated_vapour_pressure
#             - self.actual_vapour_pressure
#         )

#     def compute_evaporative_demand(self, factor):
#         """Compute evaporative demand of the atmosphere (mm d-1)."""
#         self.compute_wind_coefficient()
#         self.compute_vapour_pressure_deficit()

#         def ea(factor):
#             ea = (
#                 0.26
#                 * self.vapour_pressure_deficit
#                 * (factor + self.wind_coefficient * self.wind)
#             )
#             return ea

#         self.evaporative_demand_canopy = ea(FACTOR_CANOPY)
#         self.evaporative_demand_soil = ea(FACTOR_SOIL)
#         self.evaporative_demand_water = ea(FACTOR_WATER)

#     def penman_monteith(self):
#         """Compute evapo(transpi)ration using Penman-Monteith
#         equation (mm d-1).
#         """
#         def pm(rna, ea):
#             et = (
#                 (self.slope_of_vapour_pressure_curve * rna * 0.408) +
#                 (self.psychrometric_constant * ea)
#             ) / (self.delta + self.psychrometric_constant)
#             return et

#         self.etref = pm(
#             self.net_absorbed_radiation_canopy,
#             self.evaporative_demand_canopy
#         )
#         self.esref = pm(
#             self.net_absorbed_radiation_soil,
#             self.evaporative_demand_soil
#         )
#         self.ewref = pm(
#             self.net_absorbed_radiation_water,
#             self.evaporative_demand_water
#         )

#     def dynamic(self):
#         self.compute_saturated_vapour_pressure()
#         self.compute_actual_vapour_pressure()
#         self.compute_extraterrestrial_radiation()
#         self.compute_net_absorbed_radiation()
#         self.compute_psychrometric_constant()
#         self.compute_slope_of_vapour_pressure_curve()
#         self.compute_evaporative_demand()
#         self.penman_monteith()

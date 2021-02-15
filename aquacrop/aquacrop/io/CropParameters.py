#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas
import netCDF4
import datetime as datetime
import calendar as calendar
import sqlite3
from importlib_resources import path

from hm.api import open_hmdataarray

from .utils import read_crop_parameter_from_sqlite
from . import data

from .CarbonDioxide import refconc

import aquacrop_fc


class CropParameters(object):
    def __init__(self, model):
        self.model = model
        self.config = self.model.config.CROP_PARAMETERS
        self.model.nCrop = len(self.model.domain._coords['crop'])
        # self.model.CropID = self.model.domain._coords['crop']
        self.model.CropID = self.config['crop_id']
        self.model.CalendarType = self.config['calendar_type']
        self.model.SwitchGDD = self.config['switch_gdd']
        self.model.GDDmethod = self.config['gdd_method']
        self.load_crop_parameter_database()

    def load_crop_parameter_database(self):
        with path(data, 'crop_parameter_database.sqlite3') as db_path:
            try:
                db_path = db_path.resolve()
            except FileNotFoundError:
                pass
            self.model.CropParameterDatabase = sqlite3.connect(str(db_path))

    def initial(self):
        # Read parameters from file or config file
        self.read()

        # Make a copy of arrays with planting/harvest date, which
        # can then be adjusted in the case of leap years
        self.model.PlantingDateAdj = np.copy(self.model.PlantingDate)
        self.model.HarvestDateAdj = np.copy(self.model.HarvestDate)

        # Growing season index has a farm dimension because this is
        # automatically changed to false when a crop dies, which
        # may vary depending on farm management practices
        self.model.GrowingSeasonIndex = np.require(
            np.zeros(
                (self.model.nFarm, self.model.nCrop, self.model.domain.nxy),
                dtype=np.int32
            ),
            requirements=['A', 'O', 'W', 'F']
        )
        self.model.GrowingSeasonDayOne = np.require(
            np.zeros(
                (self.model.nFarm, self.model.nCrop, self.model.domain.nxy),
                dtype=np.int32
            ),
            requirements=['A', 'O', 'W', 'F']
        )
        # Preallocate days after planting counter
        self.model.DAP = np.require(
            np.zeros(
                (self.model.nFarm, self.model.nCrop, self.model.domain.nxy),
                dtype=np.int32
            ),
            requirements=['A', 'O', 'W', 'F']
        )
        # Preallocate integer and float parameters
        int_params_to_compute = [
            'tLinSwitch', 'CanopyDevEndCD', 'CanopyDevEndCD',
            'Canopy10PctCD', 'MaxCanopyCD', 'HIstartCD', 'HIendCD',
            'YldFormCD', 'FloweringCD'
        ]
        flt_params_to_compute = [
            'CC0', 'SxTop', 'SxBot', 'fCO2', 'dHILinear', 'HIGC',
            'CanopyDevEnd', 'Canopy10Pct', 'MaxCanopy', 'HIend',
            'FloweringEnd', 'CurrentConc'
        ]
        for param in int_params_to_compute:
            vars(self.model)[param] = np.require(
                np.zeros(
                    (self.model.nFarm, self.model.nCrop, self.model.domain.nxy),
                    dtype=np.int32
                ),
                requirements=['A', 'O', 'W', 'F']
            )
        for param in flt_params_to_compute:
            vars(self.model)[param] = np.require(
                np.zeros(
                    (self.model.nFarm, self.model.nCrop, self.model.domain.nxy),
                    dtype=np.float64
                ),
                requirements=['A', 'O', 'W', 'F']
            )

        # Compute derived crop parameters and crop calendar
        self.compute_crop_parameters()
        self.compute_crop_calendar()

    def read(self):
        # PlantPop currently real in Fortran
        # ETadj should be an option provided in config
        int_crop_params = [
            'CropType', 'PlantingDate', 'HarvestDate', 'PolHeatStress',
            'PolColdStress', 'BioTempStress', 'PlantPop',
            'Determinant', 'ETadj', 'LagAer'
        ]
        flt_crop_params = [
            'Emergence', 'MaxRooting', 'Senescence', 'Maturity',
            'HIstart', 'Flowering', 'YldForm', 'Tbase', 'Tupp',
            'Tmax_up', 'Tmax_lo', 'Tmin_up', 'Tmin_lo', 'GDD_up',
            'GDD_lo', 'fshape_b', 'PctZmin', 'Zmin', 'Zmax',
            'fshape_r', 'fshape_ex', 'SxTopQ', 'SxBotQ', 'a_Tr',
            'SeedSize', 'CCmin', 'CCx', 'CDC', 'CGC', 'Kcb', 'fage',
            'WP', 'WPy', 'fsink', 'bsted', 'bface', 'HI0', 'HIini',
            'dHI_pre', 'a_HI', 'b_HI', 'dHI0', 'exc', 'MaxFlowPct',
            'p_up1', 'p_up2', 'p_up3', 'p_up4', 'p_lo1', 'p_lo2',
            'p_lo3', 'p_lo4', 'fshape_w1', 'fshape_w2', 'fshape_w3',
            'fshape_w4', 'Aer', 'beta', 'GermThr'
        ]
        crop_params_to_read = int_crop_params + flt_crop_params
        if len(crop_params_to_read) > 0:
            for param in crop_params_to_read:

                if param in int_crop_params:
                    datatype = np.int32
                else:
                    datatype = np.float64

                # 1 Try to read from configuration file:
                if param in self.config.keys():
                    # TODO
                    parameter_values = np.array(self.config[param])

                    # ensure the parameter has a farm dimension
                    if len(parameter_values.shape) == 1:
                        if len(parameter_values) == self.model.nCrop:
                            parameter_values = np.broadcast_to(
                                parameter_values, (self.model.nFarm, self.model.nCrop))

                    if (parameter_values.shape[0] == self.model.nFarm) & (parameter_values.shape[1] == self.model.nCrop):
                        vars(self.model)[param] = np.require(
                            np.broadcast_to(
                                parameter_values[:, None],
                                (self.model.nFarm, self.model.nCrop,
                                 self.model.domain.nxy)
                            ),
                            dtype=datatype,
                            requirements=['A', 'O', 'W', 'F']
                        )

                    else:
                        raise ValueError(
                            "Error reading parameter " + param
                            + " from configuration file: length"
                            + " of parameter list must equal number"
                            + " of crops in simulation"
                        )
                else:
                    # 2 Try to read from netCDF file
                    try:
                        arr = open_hmdataarray(
                            self.model.config.CROP_PARAMETERS['filename'],
                            param,
                            self.model.domain,
                            self.model.config.CROP_PARAMETERS['is_1d'],
                            self.model.config.CROP_PARAMETERS['xy_dimname'],
                            sample=self.model.currentSampleNumber
                        )
                        vars(self.model)[param] = np.require(
                            np.broadcast_to(
                                arr.values,
                                (self.model.nFarm, self.model.nCrop,
                                 self.model.domain.nxy)
                            ),
                            dtype=datatype,
                            requirements=['A', 'O', 'W', 'F']
                        )

                    # 3 - Read from default parameter database
                    except:
                        try:
                            parameter_values = np.zeros((self.model.nCrop))
                            for index, crop_id in enumerate(self.model.CropID):
                                parameter_values[index] = read_crop_parameter_from_sqlite(
                                    self.model.CropParameterDatabase,
                                    crop_id,
                                    param
                                )[0]

                            vars(self.model)[param] = np.require(
                                np.broadcast_to(
                                    parameter_values[None, :, None],
                                    (self.model.nFarm, self.model.nCrop,
                                     self.model.domain.nxy)
                                ),
                                dtype=datatype,
                                requirements=['A', 'O', 'W', 'F']
                            )

                        except:
                            raise KeyError(
                                "Error reading parameter "
                                + param + " from crop parameter database"
                            )

    def compute_crop_parameters(self):
        aquacrop_fc.crop_parameters_w.compute_crop_parameters_w(
            self.model.CC0,
            self.model.SxTop,
            self.model.SxBot,
            self.model.PlantingDateAdj,
            self.model.HarvestDateAdj,
            self.model.CanopyDevEnd,
            self.model.Canopy10Pct,
            self.model.MaxCanopy,
            self.model.HIend,
            self.model.FloweringEnd,
            self.model.FloweringCD,
            self.model.HIGC,
            self.model.tLinSwitch,
            self.model.dHILinear,
            self.model.PlantPop,
            self.model.SeedSize,
            self.model.SxTopQ,
            self.model.SxBotQ,
            self.model.PlantingDate,
            self.model.HarvestDate,
            int(self.model.time.doy),
            int(self.model.time.timestep),
            int(self.model.time.is_leap_year),
            self.model.Senescence,
            self.model.HIstart,
            self.model.Flowering,
            self.model.Determinant,
            self.model.Emergence,
            self.model.CGC,
            self.model.CCx,
            self.model.YldForm,
            self.model.CropType,
            self.model.YldFormCD,
            self.model.HI0,
            self.model.HIini,
            self.model.nFarm,
            self.model.nCrop,
            self.model.domain.nxy
        )

    def compute_time_slice(self):
        sd = self.model.time.curr_time.timetuple().tm_yday
        hd = np.copy(self.model.HarvestDateAdj)
        hd[hd < self.model.PlantingDateAdj] += 365
        hd[sd > self.model.PlantingDateAdj] = 0
        max_harvest_date = int(np.max(hd))
        time_slc = slice(
            self.model.time.curr_time,
            self.model.time.curr_time +
            # TODO: '+1' is new addition to match Fortran code
            datetime.timedelta(int(max_harvest_date - sd + 1))
        )
        return time_slc

    def compute_crop_calendar_type_1(self):
        self.model.MaxCanopyCD = np.copy(self.model.MaxCanopy)
        self.model.CanopyDevEndCD = np.copy(self.model.CanopyDevEnd)
        self.model.HIstartCD = np.copy(self.model.HIstart)
        self.model.HIendCD = np.copy(self.model.HIend)
        self.model.YldFormCD = np.copy(self.model.YldForm)
        self.model.FloweringCD = np.copy(self.model.Flowering)

        if self.model.SwitchGDD:

            # !!!!!!!!
            #
            # UNTESTED
            #
            # !!!!!!!!

            # objective here is to define a time slice over which
            # to compute the parameters, since we need tmin/tmax
            # values to calculate growing degree days
            time_slc = self.compute_time_slice()
            self.model.tmin.select(time_slc)
            self.model.tmax.select(time_slc)
            aquacrop_w.crop_parameters_w.switch_gdd_w(
                self.model.tmin.values,
                self.model.tmax.values,
                self.model.PlantingDayAdj,
                self.model.HarvestDateAdj,
                int(self.model.time.doy),
                self.model.Emergence,
                self.model.Canopy10Pct,
                self.model.MaxRooting,
                self.model.Senescence,
                self.model.Maturity,
                self.model.MaxCanopy,
                self.model.CanopyDevEnd,
                self.model.HIstart,
                self.model.HIend,
                self.model.YldForm,
                self.model.MaxCanopyCD,
                self.model.CanopyDevEndCD,
                self.model.HIstartCD,
                self.model.HIendCD,
                self.model.YldFormCD,
                self.model.FloweringEnd,
                self.model.Flowering,
                self.model.CC0,
                self.model.CCx,
                self.model.CGC,
                self.model.CDC,
                self.model.Tupp,
                self.model.Tbase,
                self.model.CropType,
                self.model.GDDmethod,
                self.model.CalendarType,
                int(self.model.tmin.values.shape[0]),
                self.model.nCrop,
                self.model.domain.nxy
            )

            # return tmin/tmax to current time
            self.model.tmin.select(time=self.model.time.curr_time)
            self.model.tmax.select(time=self.model.time.curr_time)

    def compute_crop_calendar_type_2(self, update=False):

        time_slc = self.compute_time_slice()
        self.model.tmin.select(time_slc)
        self.model.tmax.select(time_slc)

        aquacrop_fc.crop_parameters_w.compute_crop_calendar_type2_w(
            self.model.tmin.values,
            self.model.tmax.values,
            self.model.PlantingDateAdj,
            self.model.HarvestDateAdj,
            int(self.model.time.doy),
            self.model.MaxCanopy,
            self.model.CanopyDevEnd,
            self.model.HIstart,
            self.model.HIend,
            self.model.MaxCanopyCD,
            self.model.CanopyDevEndCD,
            self.model.HIstartCD,
            self.model.HIendCD,
            self.model.YldFormCD,
            self.model.FloweringCD,
            self.model.FloweringEnd,
            self.model.Tupp,
            self.model.Tbase,
            self.model.CropType,
            self.model.GDDmethod,
            self.model.CalendarType,
            self.model.GrowingSeasonDayOne,
            int(self.model.time.timestep == 0),
            int(self.model.tmin.values.shape[0]),
            self.model.nFarm,
            self.model.nCrop,
            self.model.domain.nxy
        )

        # return tmin/tmax to current time
        # TODO: method='nearest' is new addition - CHECK
        self.model.tmin.select(
            time=self.model.time.curr_time, method='nearest')
        # TODO: method='nearest' is new addition - CHECK
        self.model.tmax.select(
            time=self.model.time.curr_time, method='nearest')

    def compute_crop_calendar(self):
        if self.model.CalendarType == 1:
            self.compute_crop_calendar_type_1()
        elif self.model.CalendarType == 2:
            self.compute_crop_calendar_type_2()

    def update_crop_parameters(self):
        if (self.model.CalendarType == 2):
            if (np.any(self.model.GrowingSeasonDayOne)):
                self.compute_crop_calendar_type_2(update=True)
                aquacrop_fc.crop_parameters_w.update_crop_parameters_w(
                    self.model.HIGC,
                    self.model.tLinSwitch,
                    self.model.dHILinear,
                    self.model.YldFormCD,
                    self.model.HI0,
                    self.model.HIini,
                    self.model.nFarm,
                    self.model.nCrop,
                    self.model.domain.nxy
                )

    def compute_water_productivity_adjustment_factor(self):
        aquacrop_fc.crop_parameters_w.compute_wp_adj_factor_w(
            self.model.fCO2,
            self.model.CurrentConc,
            self.model.conc.values,
            refconc,
            self.model.bsted,
            self.model.bface,
            self.model.fsink,
            self.model.WP,
            self.model.GrowingSeasonDayOne,
            self.model.nFarm,
            self.model.nCrop,
            self.model.domain.nxy
        )

    def dynamic(self):
        self.adjust_planting_and_harvesting_date()
        self.update_growing_season()
        self.compute_water_productivity_adjustment_factor()
        self.update_crop_parameters()

    def update_growing_season(self):
        endnum = netCDF4.date2num(
            datetime.datetime(
                self.model.time.endtime.year,
                self.model.time.endtime.month,
                self.model.time.endtime.day
            ),
            units='days since 1900-01-01 00:00:00'
        )
        startnum = netCDF4.date2num(
            datetime.datetime(
                self.model.time.year, 1, 1
            ),
            units='days since 1900-01-01 00:00:00'
        )
        aquacrop_fc.crop_parameters_w.update_growing_season_w(
            self.model.GrowingSeasonIndex,
            self.model.GrowingSeasonDayOne,
            self.model.DAP,
            self.model.PlantingDateAdj,
            self.model.HarvestDateAdj,
            self.model.CropDead,
            self.model.CropMature,
            self.model.time.doy,
            self.model.time.timestep,
            int(startnum),
            int(endnum),
            self.model.nFarm,
            self.model.nCrop,
            self.model.domain.nxy
        )

    def adjust_planting_and_harvesting_date(self):
        aquacrop_fc.crop_parameters_w.adjust_pd_hd_w(
            self.model.PlantingDateAdj,
            self.model.HarvestDateAdj,
            self.model.PlantingDate,
            self.model.HarvestDate,
            self.model.time.doy,
            self.model.time.timestep,
            int(self.model.time.is_leap_year),
            self.model.nFarm,
            self.model.nCrop,
            self.model.domain.nxy
        )

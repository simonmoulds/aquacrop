#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas
import netCDF4 as nc
import datetime as datetime
import calendar as calendar
import sqlite3
from importlib_resources import path

from hm import file_handling
from hm.Messages import ModelError
from . import data

import aquacrop_fc

class CropParameters(object):
    def __init__(self, CropParameters_variable):
        self.var = CropParameters_variable
        self.get_num_crop()
        self.get_crop_id()
        self.load_crop_parameter_database()
        
    def get_num_crop(self):
        pass

    def get_crop_id(self):
        ids = str(self.var._configuration.CROP_PARAMETERS['cropID'])
        try:
            ids = [int(x) for x in ids.split(',')]
            self.var.CropID = ids
        except:
            self.var.CropID = None

    def initial(self):
        self.read()        
        self.var.PlantingDateAdj = np.copy(self.var.PlantingDate)
        self.var.HarvestDateAdj = np.copy(self.var.HarvestDate)
        
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.GrowingSeasonIndex = np.copy(arr_zeros.astype(bool))
        self.var.GrowingSeasonDayOne = np.copy(arr_zeros.astype(bool))
        int_params_to_compute = [
            'tLinSwitch','DAP','CanopyDevEndCD','CanopyDevEndCD',
            'Canopy10PctCD','MaxCanopyCD','HIstartCD','HIendCD',
            'YldFormCD','FloweringCD'
        ]
        flt_params_to_compute = [
            'CC0','SxTop','SxBot','fCO2','dHILinear','HIGC',
            'CanopyDevEnd',
            'Canopy10Pct',
            'MaxCanopy',
            'HIend',
            'FloweringEnd',
            'CurrentConc'
        ]
        for param in int_params_to_compute:
            vars(self.var)[param] = np.copy(arr_zeros.astype(np.int32))

        for param in flt_params_to_compute:
            vars(self.var)[param] = np.copy(arr_zeros.astype(np.float64))

        self.compute_crop_parameters()

    def get_crop_parameter_names(self):        
        self.var.crop_parameters_to_read = [
            'CropType','PlantingDate','HarvestDate','Emergence','MaxRooting',
            'Senescence','Maturity','HIstart','Flowering','YldForm',
            'PolHeatStress','PolColdStress','BioTempStress','PlantPop',
            'Determinant','ETadj','LagAer','Tbase','Tupp','Tmax_up','Tmax_lo',
            'Tmin_up','Tmin_lo','GDD_up','GDD_lo','fshape_b','PctZmin','Zmin',
            'Zmax','fshape_r','fshape_ex','SxTopQ','SxBotQ','a_Tr','SeedSize',
            'CCmin','CCx','CDC','CGC','Kcb','fage','WP','WPy','fsink','bsted',
            'bface','HI0','HIini','dHI_pre','a_HI','b_HI','dHI0','exc',
            'MaxFlowPct','p_up1','p_up2','p_up3','p_up4','p_lo1','p_lo2','p_lo3',
            'p_lo4','fshape_w1','fshape_w2','fshape_w3','fshape_w4','Aer','beta',
            'GermThr']

    def load_crop_parameter_database(self):
        with path(data, 'crop_parameter_database.sqlite3') as db_path:
            try:
                db_path = db_path.resolve()
            except FileNotFoundError:
                pass
            self.var.CropParameterDatabase = sqlite3.connect(str(db_path))

    def adjust_planting_and_harvesting_date(self):
        leap_year = calendar.isleap(self.var._modelTime.currTime.year)
        aquacrop_fc.crop_parameters_w.adjust_pd_hd_w(
            np.int32(self.var.PlantingDateAdj).T,
            np.int32(self.var.HarvestDateAdj).T,
            np.int32(self.var.PlantingDate).T,
            np.int32(self.var.HarvestDate).T,
            np.int32(self.var._modelTime.doy),
            np.int32(self.var._modelTime.timeStepPCR),
            np.int32(leap_year),
            self.var.nFarm, self.var.nCrop, self.var.nCell
        )

    def update_growing_season(self):
        gs = np.int32(self.var.GrowingSeasonIndex.copy())
        gsd = np.int32(self.var.GrowingSeasonDayOne.copy())
        aquacrop_fc.crop_parameters_w.update_growing_season_w(
            gs.T,
            gsd.T,
            # np.int32(self.var.GrowingSeasonIndex).T,
            # np.int32(self.var.GrowingSeasonDayOne).T,
            np.int32(self.var.DAP).T,
            np.int32(self.var.PlantingDateAdj).T,
            np.int32(self.var.HarvestDateAdj).T,
            np.int32(self.var.CropDead).T,
            np.int32(self.var.CropMature).T,
            self.var._modelTime.doy,
            self.var._modelTime.timeStepPCR,
            self.var._modelTime.currentYearStartNum,
            self.var._modelTime.endTimeNum,
            self.var.nFarm, self.var.nCrop, self.var.nCell
            )
        self.var.GrowingSeasonIndex = gs.astype(bool)
        self.var.GrowingSeasonDayOne = gsd.astype(bool)
        
    def compute_crop_parameters(self):
        self.compute_initial_canopy_cover()
        self.compute_root_extraction_terms()
        self.adjust_planting_and_harvesting_date()
        self.compute_canopy_dev_end()
        self.compute_canopy_10pct()
        self.compute_max_canopy()
        self.compute_hi_end()
        self.compute_flowering_end_cd()                
        self.compute_crop_calendar()
        self.compute_HIGC()
        self.compute_HI_linear()
        
    def compute_initial_canopy_cover(self):
        self.var.CC0 = np.round(10000. * (self.var.PlantPop * self.var.SeedSize) * 10 ** -8) / 10000
        
    def compute_root_extraction_terms(self):
        aquacrop_fc.crop_parameters_w.compute_root_extraction_terms_w(
            self.var.SxTop.T,
            self.var.SxBot.T,
            self.var.SxTopQ.T,
            self.var.SxBotQ.T,
            self.var.nFarm, self.var.nCrop, self.var.nCell
            )
        
    def compute_HI_linear(self):
        aquacrop_fc.crop_parameters_w.compute_hi_linear_w(
            self.var.tLinSwitch.T,
            self.var.dHILinear.T,
            self.var.HIini.T,
            self.var.HI0.T,
            self.var.HIGC.T,
            self.var.YldFormCD.T,
            self.var.nFarm, self.var.nCrop, self.var.nCell
        )
                    
    def compute_HIGC(self):
        aquacrop_fc.crop_parameters_w.compute_higc_w(
            self.var.HIGC.T,
            self.var.YldFormCD.T,
            self.var.HI0.T,
            self.var.HIini.T,
            self.var.nFarm, self.var.nCrop, self.var.nCell
            )

    def compute_canopy_dev_end(self):
        # Time from sowing to end of vegetative growth period
        self.var.CanopyDevEnd = np.copy(self.var.Senescence)
        cond1 = (self.var.Determinant == 1)
        self.var.CanopyDevEnd[cond1] = (np.round(self.var.HIstart + (self.var.Flowering / 2)))[cond1]
        
    def compute_canopy_10pct(self):
        # Time from sowing to 10% canopy cover (non-stressed conditions)
        self.var.Canopy10Pct = np.round(
            self.var.Emergence +
            np.divide(
                np.log(
                    np.divide(
                        0.1,
                        self.var.CC0,
                        out=np.ones_like(self.var.CC0),
                        where=self.var.CC0!=0
                        )
                ),
                self.var.CGC,
                out=np.zeros_like(self.var.CGC),
                where=self.var.CGC!=0
            )
        )

    def compute_max_canopy(self):
        # Time from sowing to maximum canopy cover (non-stressed conditions)
        self.var.MaxCanopy = np.round(
            self.var.Emergence +
            (np.log(
                (0.25 * self.var.CCx * self.var.CCx / self.var.CC0)
                / (self.var.CCx - (0.98 * self.var.CCx))
            ) /
             self.var.CGC)
        )

    def compute_hi_end(self):
        # Time from sowing to end of yield formation
        self.var.HIend = self.var.HIstart + self.var.YldForm

    def compute_flowering_end_cd(self):
        arr_zeros = np.zeros_like(self.var.CropType)
        self.var.FloweringEnd = np.copy(arr_zeros)
        FloweringEndCD = np.copy(arr_zeros)
        self.var.FloweringCD = np.copy(arr_zeros)
        cond2 = (self.var.CropType == 3)
        self.var.FloweringEnd[cond2] = (self.var.HIstart + self.var.Flowering)[cond2]
        FloweringEndCD[cond2] = self.var.FloweringEnd[cond2]
        self.var.FloweringCD[cond2] = self.var.Flowering[cond2]

    def compute_pd_hd(self):
        pd = np.copy(self.var.PlantingDateAdj)
        hd = np.copy(self.var.HarvestDateAdj)
        hd[hd < pd] += 365
        sd = self.var._modelTime.currTime.timetuple().tm_yday
        planting_day_before_start_day = sd > pd
        pd[planting_day_before_start_day] = 0  # is this possible?
        hd[planting_day_before_start_day] = 0
        return pd,hd

    def compute_day_index(self, pd, hd):
        sd = self.var._modelTime.currTime.timetuple().tm_yday
        max_harvest_date = int(np.max(hd))
        day_index = np.arange(sd, max_harvest_date + 1)
        day_index = day_index[:,None,None,None] * np.ones((self.var.nFarm, self.var.nCrop, self.var.nCell))[None,...]
        return day_index
    
    def compute_growing_season_index(self, day_idx, pd, hd):
        growing_season_index = ((day_idx >= pd) & (day_idx <= hd))
        return growing_season_index
        
    def compute_cumulative_gdd(self, hd, growing_season_index):        
        sd = self.var._modelTime.currTime.timetuple().tm_yday
        max_harvest_date = int(np.max(hd))
        start_time = self.var._modelTime.currTime

        tmin = file_handling.netcdf_time_slice_to_array(
            self.var.weather.minDailyTemperatureNC,
            self.var.weather.tminVarName,
            start_time,
            start_time + datetime.timedelta(int(max_harvest_date - sd)),
            cloneMapFileName = self.var.cloneMapFileName,
            LatitudeLongitude = True)

        tmax = file_handling.netcdf_time_slice_to_array(
            self.var.weather.maxDailyTemperatureNC,
            self.var.weather.tmaxVarName,
            start_time,
            start_time + datetime.timedelta(int(max_harvest_date - sd)),
            cloneMapFileName = self.var.cloneMapFileName,
            LatitudeLongitude = True)

        # broadcast to crop dimension
        tmin = tmin[...,self.var.landmask]
        tmax = tmax[...,self.var.landmask]
        tmin = tmin[:,None,None,...] * np.ones_like(growing_season_index)
        tmax = tmax[:,None,None,...] * np.ones_like(growing_season_index)

        # for convenience
        tupp = np.broadcast_to(self.var.Tupp, tmin.shape).copy()
        tbase = np.broadcast_to(self.var.Tbase, tmin.shape).copy()

        # calculate GDD according to the various methods
        if self.var.GDDmethod == 1:
            tmean = ((tmax + tmin) / 2)
            tmean = np.clip(tmean, self.var.Tbase, self.var.Tupp)
        elif self.var.GDDmethod == 2:
            tmax = np.clip(tmax, self.var.Tbase, self.var.Tupp)
            tmin = np.clip(tmin, self.var.Tbase, self.var.Tupp)
            tmean = ((tmax + tmin) / 2)
        elif self.var.GDDmethod == 3:
            tmax = np.clip(tmax, self.var.Tbase, self.var.Tupp)
            tmin = np.clip(tmin, None, self.var.Tupp)
            tmean = ((tmax + tmin) / 2)
            tmean = np.clip(tmean, self.var.Tbase, None)

        tmean *= growing_season_index
        tbase *= growing_season_index
        GDD = (tmean - tbase)
        GDDcum = np.cumsum(GDD, axis=0)
        return GDDcum

    def compute_crop_calendar_type_1(self):        
        EmergenceCD = np.copy(self.var.Emergence)
        Canopy10PctCD = np.copy(self.var.Canopy10Pct)
        MaxRootingCD = np.copy(self.var.MaxRooting)
        SenescenceCD = np.copy(self.var.Senescence)
        MaturityCD = np.copy(self.var.Maturity)
        self.var.MaxCanopyCD = np.copy(self.var.MaxCanopy)
        self.var.CanopyDevEndCD = np.copy(self.var.CanopyDevEnd)
        self.var.HIstartCD = np.copy(self.var.HIstart)
        self.var.HIendCD = np.copy(self.var.HIend)
        self.var.YldFormCD = np.copy(self.var.YldForm)            
        FloweringEndCD = np.copy(self.var.FloweringEnd)
        self.var.FloweringCD = np.copy(self.var.Flowering)

        if self.var.SwitchGDD:                
            pd, hd = self.compute_pd_hd()
            day_idx = self.compute_day_index(pd, hd)
            growing_season_idx = self.compute_growing_season_index(day_idx, pd, hd)            
            GDDcum = self.compute_cumulative_gdd(hd, growing_season_idx)
            if (self.var.CalendarType == 1) & (self.var.SwitchGDD):
                # Find GDD equivalent for each crop calendar variable
                m, n, p = pd.shape
                I, J, K = np.ogrid[:m,:n,:p]
                emergence_idx = pd + EmergenceCD
                self.var.Emergence = GDDcum[emergence_idx,I,J,K]
                canopy10pct_idx = pd + Canopy10PctCD
                self.var.Canopy10Pct = GDDcum[canopy10pct_idx,I,J,K]
                maxrooting_idx = pd + MaxRootingCD
                self.var.MaxRooting = GDDcum[maxrooting_idx,I,J,K]
                maxcanopy_idx = pd + self.var.MaxCanopyCD
                self.var.MaxCanopy = GDDcum[maxcanopy_idx,I,J,K]
                canopydevend_idx = pd + self.var.CanopyDevEndCD
                self.var.CanopyDevEnd = GDDcum[canopydevend_idx,I,J,K]
                senescence_idx = pd + SenescenceCD
                self.var.Senescence = GDDcum[senescence_idx,I,J,K]
                maturity_idx = pd + MaturityCD
                self.var.Maturity = GDDcum[maturity_idx,I,J,K]
                histart_idx = pd + self.var.HIstartCD
                self.var.HIstart = GDDcum[histart_idx,I,J,K]
                hiend_idx = pd + self.var.HIendCD
                self.var.HIend = GDDcum[hiend_idx,I,J,K]
                yldform_idx = pd + self.var.YldFormCD
                self.var.YldForm = GDDcum[yldform_idx,I,J,K]

                cond2 = (self.var.CropType == 3)
                floweringend_idx = pd + FloweringEndCD
                self.var.FloweringEnd[cond2] = GDDcum[floweringend_idx,I,J,K][cond2]
                self.var.Flowering[cond2] = (self.var.FloweringEnd - self.var.HIstart)[cond2]

                # Convert CGC to GDD mode
                self.var.CGC = (np.log((((0.98 * self.var.CCx) - self.var.CCx) * self.var.CC0) / (-0.25 * (self.var.CCx ** 2)))) / (-(self.var.MaxCanopy - self.var.Emergence))

                # Convert CDC to GDD mode
                tCD = MaturityCD - SenescenceCD
                tCD[tCD <= 0] = 1
                tGDD = self.var.Maturity - self.var.Senescence
                tGDD[tGDD <= 0] = 5
                self.var.CDC = (self.var.CCx / tGDD) * np.log(1 + ((1 - self.var.CCi / self.var.CCx) / 0.05))

                # Set calendar type to GDD mode
                self.var._configuration.CROP_PARAMETERS['CalendarType'] = "2"

    def compute_crop_calendar_type_2(self, update=False):
        pd, hd = self.compute_pd_hd()
        day_idx = self.compute_day_index(pd, hd)
        growing_season_idx = self.compute_growing_season_index(day_idx, pd, hd)            
        GDDcum = self.compute_cumulative_gdd(hd, growing_season_idx)
        
        maxcanopy_idx = np.copy(day_idx)
        maxcanopy_idx[np.logical_not(GDDcum > self.var.MaxCanopy)] = 999
        maxcanopy_idx = np.nanmin(maxcanopy_idx, axis=0)

        canopydevend_idx = np.copy(day_idx)
        canopydevend_idx[np.logical_not(GDDcum > self.var.CanopyDevEnd)] = 999        
        canopydevend_idx = np.nanmin(canopydevend_idx, axis=0)

        histart_idx = np.copy(day_idx)
        histart_idx[np.logical_not(GDDcum > self.var.HIstart)] = 999
        histart_idx = np.nanmin(histart_idx, axis=0)

        hiend_idx = np.copy(day_idx)
        hiend_idx[np.logical_not(GDDcum > self.var.HIend)] = 999
        hiend_idx = np.nanmin(hiend_idx, axis=0)

        floweringend_idx = np.copy(day_idx)
        floweringend_idx[np.logical_not(GDDcum > self.var.FloweringEnd)] = 999
        floweringend_idx = np.nanmin(floweringend_idx, axis=0)

        if update:
            maxcanopycd = maxcanopy_idx - pd + 1
            self.var.MaxCanopyCD[self.var.GrowingSeasonDayOne] = maxcanopycd[self.var.GrowingSeasonDayOne]
            canopydevendcd = canopydevend_idx - pd + 1
            self.var.CanopyDevEndCD[self.var.GrowingSeasonDayOne] = canopydevendcd[self.var.GrowingSeasonDayOne]
            histartcd = histart_idx - pd + 1
            self.var.HIstartCD[self.var.GrowingSeasonDayOne] = histartcd[self.var.GrowingSeasonDayOne]
            hiendcd = hiend_idx - pd + 1
            self.var.HIendCD[self.var.GrowingSeasonDayOne] = hiendcd[self.var.GrowingSeasonDayOne]
            floweringendcd = (floweringend_idx - pd + 1) - self.var.HIstartCD
            cond1 = (self.var.CropType == 3) & (self.var.GrowingSeasonDayOne)
            self.var.FloweringCD[cond1] = floweringendcd[cond1]
            yldformcd = self.var.HIendCD - self.var.HIstartCD
            self.var.YldFormCD[self.var.GrowingSeasonDayOne] = yldformcd[self.var.GrowingSeasonDayOne]
            
        else:            
            self.var.MaxCanopyCD = maxcanopy_idx - pd + 1
            self.var.CanopyDevEndCD = canopydevend_idx - pd + 1
            self.var.HIstartCD = histart_idx - pd + 1
            self.var.HIendCD = hiend_idx - pd + 1        
            cond1 = (self.var.CropType == 3)
            floweringendcd = (floweringend_idx - pd + 1) - self.var.HIstartCD
            self.var.FloweringCD[cond1] = floweringendcd[cond1]        
            self.var.YldFormCD = self.var.HIendCD - self.var.HIstartCD
        print(self.var.CanopyDevEndCD)
        
    def compute_crop_calendar(self):       
        if self.var.CalendarType == 1:
            self.compute_crop_calendar_type_1()
        elif self.var.CalendarType == 2:
            self.compute_crop_calendar_type_2()
            
    def update_crop_parameters(self):
        if (self.var.CalendarType == 2):
            if (np.any(self.var.GrowingSeasonDayOne)):
                self.compute_crop_calendar_type_2(update=True)
                self.compute_HIGC()
                self.compute_HI_linear()

    def compute_water_productivity_adjustment_factor(self):
        """Function to calculate water productivity adjustment factor 
        for elevation in CO2 concentration"""

        # Get CO2 weighting factor
        fw = np.zeros_like(self.var.conc)
        cond1 = (self.var.conc > self.var.RefConc)
        cond11 = (cond1 & (self.var.conc >= 550))
        fw[cond11] = 1
        cond12 = (cond1 & np.logical_not(cond11))
        fw[cond12] = (1 - ((550 - self.var.conc) / (550 - self.var.RefConc)))[cond12]

        # Determine adjustment for each crop in first year of simulation
        fCO2 = ((self.var.conc / self.var.RefConc) /
                (1 + (self.var.conc - self.var.RefConc) * ((1 - fw)
                                           * self.var.bsted + fw
                                           * ((self.var.bsted * self.var.fsink)
                                              + (self.var.bface
                                                 * (1 - self.var.fsink))))))

        # Consider crop type
        ftype = (40 - self.var.WP) / (40 - 20)
        ftype = np.clip(ftype, 0, 1)
        fCO2 = 1 + ftype * (fCO2 - 1)
        
        self.var.fCO2[self.var.GrowingSeasonDayOne] = fCO2[self.var.GrowingSeasonDayOne]
        conc = np.broadcast_to(self.var.conc, (self.var.nFarm, self.var.nCrop, self.var.nCell))
        # conc = (self.var.conc[None,:] * np.ones((self.var.nCrop))[:,None])
        self.var.CurrentConc[self.var.GrowingSeasonDayOne] = conc[self.var.GrowingSeasonDayOne]
        
    def dynamic(self):
        """Function to update parameters for current crop grown as well 
        as counters pertaining to crop growth
        """
        # Update crop parameters for currently grown crops
        # self.compute_water_productivity_adjustment_factor()
        self.adjust_planting_and_harvesting_date()
        self.update_growing_season()
        self.compute_water_productivity_adjustment_factor()
        # self.read_crop_area()   # TEST
        self.update_crop_parameters()
                                    
def read_params(fn):
    with open(fn) as f:
        content = f.read().splitlines()

    # remove commented lines
    content = [x for x in content if re.search('^(?!%%).*', x)]
    content = [re.split('\s*:\s*', x) for x in content]
    params = {}
    for x in content:
        if len(x) > 1:
            nm = x[0]
            val = x[1]
            params[nm] = val
    return params

class CropParametersPoint(CropParameters):
    def __init__(self, CropParameters_variable):
        super(CropParametersPoint, self).__init__(CropParameters_variable)

    def get_num_crop(self):
        self.var.nCrop = 1

    def read(self):
        self.get_crop_parameter_names()        
        crop_parameter_values = read_params(self.var._configuration.CROP_PARAMETERS['cropParametersFile'])
        for param in self.var.crop_parameters_to_read:
            read_from_file = (param in crop_parameter_values.keys())
            if read_from_file:
                d = crop_parameter_values[param]
                d = np.broadcast_to(d[None,None,:], (self.var.nFarm,self.var.nCrop, self.var.nCell))
                vars(self.var)[param] = d.copy()                
            else:                
                try:
                    parameter_values = np.zeros((self.var.nCrop))
                    for index,crop_id in enumerate(self.var.CropID):
                        parameter_values[index] = file_handling.read_crop_parameter_from_sqlite(
                            self.var.CropParameterDatabase,
                            crop_id,
                            param
                        )[0]
                    vars(self.var)[param] = np.broadcast_to(
                        parameter_values[:,None,None],
                        (self.var.nFarm, self.var.nCrop, self.var.nCell)
                    )
                except:
                    raise ModelError("Error reading parameter " + param + " from crop parameter database")
        
class CropParametersGrid(CropParameters):    
    def __init__(self, CropParameters_variable):
        super(CropParametersGrid, self).__init__(CropParameters_variable)
        self.var.CalendarType = int(self.var._configuration.CROP_PARAMETERS['CalendarType'])
        self.var.SwitchGDD = bool(int(self.var._configuration.CROP_PARAMETERS['SwitchGDD']))
        self.var.GDDmethod = int(self.var._configuration.CROP_PARAMETERS['GDDmethod'])                                      

    def get_num_crop(self):
        self.var.nCrop = file_handling.get_dimension_variable(
            self.var._configuration.CROP_PARAMETERS['cropParametersNC'],
            'crop'
        ).size
        
    def read(self):        
        self.get_crop_parameter_names()
        if len(self.var.crop_parameters_to_read) > 0:
            for param in self.var.crop_parameters_to_read:
                read_from_netcdf = file_handling.check_if_nc_has_variable(
                    self.var._configuration.CROP_PARAMETERS['cropParametersNC'],
                    param
                    )
                if read_from_netcdf:
                    d = file_handling.netcdf_to_arrayWithoutTime(
                        self.var._configuration.CROP_PARAMETERS['cropParametersNC'],
                        param,
                        cloneMapFileName=self.var.cloneMapFileName)
                    d = d[self.var.landmask_crop].reshape(self.var.nCrop,self.var.nCell)
                    vars(self.var)[param] = np.broadcast_to(d, (self.var.nFarm, self.var.nCrop, self.var.nCell))                    
                else:
                    try:
                        parameter_values = np.zeros((self.var.nCrop))
                        for index,crop_id in enumerate(self.var.CropID):
                            parameter_values[index] = file_handling.read_crop_parameter_from_sqlite(
                                self.var.CropParameterDatabase,
                                crop_id,
                                param
                            )[0]
                        vars(self.var)[param] = np.broadcast_to(
                            parameter_values[:,None,None],
                            (self.var.nFarm, self.var.nCrop, self.var.nCell)
                        )
                        
                    except:
                        raise ModelError("Error reading parameter " + param + " from crop parameter database")

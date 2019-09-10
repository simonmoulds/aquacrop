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

class CropArea(object):
    
    def __init__(self, CropArea_variable):
        self.var = CropArea_variable
        self.var.CropAreaFileNC = str(self.var._configuration.CROP_PARAMETERS['cropAreaNC'])
        self.var.CropAreaVarName = str(self.var._configuration.CROP_PARAMETERS['cropAreaVarName'])
        self.var.CroplandAreaFileNC = str(self.var._configuration.CROP_PARAMETERS['croplandAreaNC'])
        self.var.CroplandAreaVarName = str(self.var._configuration.CROP_PARAMETERS['croplandAreaVarName'])
        self.var.AnnualChangeInCropArea = bool(int(self.var._configuration.CROP_PARAMETERS['AnnualChangeInCropArea']))
        self.var.landmask_crop = np.broadcast_to(
            self.var.landmask[None,:,:],
            (self.var.nCrop, self.var.nLat, self.var.nLon))
        self.var.landmask_farm_crop = np.broadcast_to(
            self.var.landmask[None,None,:,:],
            (self.var.nFarm, self.var.nCrop, self.var.nLat, self.var.nLon))
        
    def initial(self):
        self.var.CurrentCropArea = np.ones((self.var.nFarm, self.var.nCrop, self.var.nCell))
        # test f2py integration:
        # arr=np.random.rand(100,100,100,100)
        # tot=aquacrop_fc.f2py_run(arr.T)
        # print(tot)
        # nx=100
        # ny=100
        # A=np.random.randint(0,10,(nx,ny))
        # B=np.random.randint(0,10,(nx,ny))
        # # C=np.zeros_like(A, dtype='float64')
        # C,D=aquacrop_fc.evap_layer_water_content(A.T,B.T,nx,ny)
        # print(A[0,0])
        # print(B[0,0])
        # print(C[0,0])              
        # print(D[0,0])              
        
    def read_cropland_area(self):
        if self.var.AnnualChangeInCropArea:
            if self.var._modelTime.timeStepPCR == 1 or self.var._modelTime.doy == 1:
                date = '%04i-%02i-%02i' % (self.var._modelTime.year, 1, 1)
                CroplandArea = file_handling.netcdf_to_array(
                    self.var.CroplandAreaFileNC,
                    # 'cropland_area',
                    self.var.CroplandAreaVarName,
                    date,
                    useDoy = None,
                    cloneMapFileName = self.var.cloneMapFileName,
                    LatitudeLongitude = True)
                self.var.CroplandArea = CroplandArea[self.var.landmask]
                
        else:
            if self.var._modelTime.timeStepPCR == 1:
                if not self.var.CropAreaFileNC == "None":
                    CroplandArea = file_handling.netcdf_to_arrayWithoutTime(
                        self.var.CroplandAreaFileNC,
                        # 'cropland_area',
                        self.var.CroplandAreaVarName,
                        cloneMapFileName = self.var.cloneMapFileName)
                    self.var.CroplandArea = CroplandArea[self.var.landmask]
                    
                else:
                    self.var.CroplandArea = np.ones((self.var.nCell)) * self.var.nCrop
    
        self.var.CroplandArea = np.broadcast_to(
            self.var.CroplandArea, (
                self.var.nFarm,
                self.var.nCrop,
                self.var.nCell))
        self.var.CroplandArea = np.float64(self.var.CroplandArea)
        
    def read_crop_area(self, date = None):
        if date is None:
            crop_area = file_handling.netcdf_to_arrayWithoutTime(
                self.var.CropAreaFileNC,
                self.var.CropAreaVarName,
                cloneMapFileName = self.var.cloneMapFileName,
                LatitudeLongitude = True)
        else:
            crop_area = file_handling.netcdf_to_array(
                self.var.CropAreaFileNC,
                self.var.CropAreaVarName,
                date,
                useDoy = None,
                cloneMapFileName = self.var.cloneMapFileName,
                LatitudeLongitude = True)
            
        crop_area_has_farm_dimension = (
            file_handling.check_if_nc_variable_has_dimension(
                self.var.CropAreaFileNC,
                self.var.CropAreaVarName,
                'farm'))

        if crop_area_has_farm_dimension:
            crop_area = np.reshape(
                crop_area[self.var.landmask_farm_crop],
                (self.var.nFarm, self.var.nCrop, self.var.nCell))
        else:
            crop_area = np.reshape(
                crop_area[self.var.landmask_crop],
                (self.var.nCrop, self.var.nCell))
            crop_area = np.broadcast_to(
                crop_area[None,:,:],
                (self.var.nFarm, self.var.nCrop, self.var.nCell))
            
        crop_area = np.float64(crop_area)
        return crop_area

    def scale_crop_area(self):
        """Function to scale crop area to match cropland area
        """
        pd = self.var.PlantingDate.copy()[0,...]
        hd = self.var.HarvestDate.copy()[0,...]
        hd[hd < pd] += 365
        max_harvest_date = int(np.max(hd))
        day_idx = np.arange(1, max_harvest_date + 1)[:,None,None] * np.ones((self.var.nCrop, self.var.nCell))[None,:,:]
        growing_season_idx = ((day_idx >= pd) & (day_idx <= hd))
        crop_area = self.var.CropArea[0,...]  # remove farm dimension
        crop_area_daily = crop_area[None,...] * growing_season_idx  # get daily crop area
        total_crop_area_daily = np.sum(crop_area_daily, axis=1)     # sum of all crops grown on a given day
        max_crop_area = np.max(total_crop_area_daily, axis=0)       # get the max crop area considering all growing seasons
        scale_factor = np.divide(self.var.CroplandArea, max_crop_area, out=np.zeros_like(self.var.CroplandArea), where=max_crop_area>0)  # compute scale factor by dividing cropland area by max crop area
        self.var.CropArea *= scale_factor
        
        # # TEST:
        # crop_area = self.var.CropArea[0,...]
        # crop_area_daily = crop_area[None,...] * growing_season_idx
        # total_crop_area_daily = np.sum(crop_area_daily, axis=1)
        # max_crop_area = np.max(total_crop_area_daily, axis=0)

    def set_crop_area(self):
        """Function to read crop area"""
        if self.var.AnnualChangeInCropArea:
            if self.var._modelTime.timeStepPCR == 1 or self.var._modelTime.doy == 1:
                # In this case crop area is updated on the first day of each year,
                # hence in order to prevent the area under a specific crop changing
                # mid-season, it is necessary to introduce an intermediate variable
                # (i.e. CropAreaNew) and only update the area on the first day of
                # the growing season.
                date = '%04i-%02i-%02i' % (self.var._modelTime.year, 1, 1)
                self.var.CropAreaNew = self.read_crop_area(date = date)
                    
            if np.any(self.var.GrowingSeasonDayOne):
                self.var.CropArea[self.var.GrowingSeasonDayOne] = self.var.CropAreaNew[self.var.GrowingSeasonDayOne]
                self.scale_crop_area()
                
        else:
            if self.var._modelTime.timeStepPCR == 1:
                if not self.var.CropAreaFileNC == "None":
                    # If crop area doesn't change then there is no need for an
                    # intermediate variable
                    self.var.CropArea = self.read_crop_area(date = None)
                else:
                    self.var.CropArea = np.ones(
                        (self.var.nFarm,
                         self.var.nCrop,
                         self.var.nCell))
                self.scale_crop_area()

    def compute_current_crop_area(self):
        """Function to work out the relative area of current crops and 
        fallow area, and divide fallow land proportionally between 
        crops that are not currently grown. This is necessary because 
        the program continues to compute the water balance for crops 
        which are not currently grown).
        """        
        crop_area = self.var.CropArea * self.var.GrowingSeasonIndex
        total_crop_area = np.sum(crop_area, axis=(0,1))

        # Compute scale factor to represent the relative area of
        # each crop not currently grown by dividing fallow area
        # by total fallow area
        crop_area_not_grown = (
            self.var.CropArea
            * np.logical_not(self.var.GrowingSeasonIndex))
        total_crop_area_not_grown = np.sum(
            crop_area_not_grown
            * np.logical_not(self.var.GrowingSeasonIndex))
        scale_factor = np.divide(
            crop_area_not_grown,
            total_crop_area_not_grown,
            out=np.zeros_like(crop_area_not_grown),
            where=total_crop_area_not_grown>0)
        
        # Compute the area which remains fallow during the growing
        # season, and scale according to the relative area of the
        # crops *not* currently grown.
        target_fallow_area = np.clip(self.var.CroplandArea - total_crop_area, 0, None)
        fallow_area = target_fallow_area * scale_factor        
        self.var.CurrentCropArea = self.var.CropArea.copy()        
        self.var.CurrentCropArea[np.logical_not(self.var.GrowingSeasonIndex)] = (
            (fallow_area[np.logical_not(self.var.GrowingSeasonIndex)]))

        # CurrentCropArea represents the crop area in the entire grid
        # cell; it does not represent the area of the respective crops
        # grown on the respective farms. Thus, create a farm scale factor
        # to scale the current cropped area down to the level of individual
        # farms.
        farm_scale_factor = np.divide(
            self.var.FarmArea,
            self.var.CroplandArea,
            out=np.zeros_like(self.var.CroplandArea),
            where=self.var.CroplandArea>0)

        # The farm scale factor represents the fraction of total cropland
        # found within each farm. Thus, multiplying the scale factor by
        # CurrentCropArea provides an estimate of the area of each crop
        # within a given farm.        
        self.var.FarmCropArea = farm_scale_factor * self.var.CurrentCropArea
        
    def dynamic(self):
        self.read_cropland_area()
        self.set_crop_area()
        self.compute_current_crop_area()

class CropParameters(object):
    
    def __init__(self, CropParameters_variable):
        self.var = CropParameters_variable

        # TODO: sort this section out - most can go to 'initial' section, I think
        
        with path(data, 'crop_parameter_database.sqlite3') as db_path:
            try:
                db_path = db_path.resolve()
            except FileNotFoundError:
                pass
            self.var.CropParameterDatabase = sqlite3.connect(str(db_path))
            
        self.var.CropParameterFileNC = str(self.var._configuration.CROP_PARAMETERS['cropParametersNC'])
        self.var.nCrop = file_handling.get_dimension_variable(self.var.CropParameterFileNC,'crop').size
        ids = str(self.var._configuration.CROP_PARAMETERS['cropID'])
        try:
            ids = [int(x) for x in ids.split(',')]
            self.var.CropID = ids
        except:
            self.var.CropID = None
        
        # TODO: move these to main model class?
        self.var.CalendarType = int(self.var._configuration.CROP_PARAMETERS['CalendarType'])
        self.var.SwitchGDD = bool(int(self.var._configuration.CROP_PARAMETERS['SwitchGDD']))
        self.var.GDDmethod = int(self.var._configuration.CROP_PARAMETERS['GDDmethod'])
                                      
        self.var.landmask_crop = (
            self.var.landmask[None,:,:]
            * np.ones((self.var.nCrop), dtype=np.bool)[:,None,None])

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.GrowingSeasonIndex = np.copy(arr_zeros.astype(bool))
        self.var.GrowingSeasonDayOne = np.copy(arr_zeros.astype(bool))
        self.var.DAP = np.copy(arr_zeros)
        # self.var.CurrentCropArea = np.ones((self.var.nFarm, self.var.nCrop, self.var.nCell))

        # # TODO: move these to main model class?
        # self.var.CalendarType = int(self.var._configuration.CROP_PARAMETERS['CalendarType'])
        # self.var.SwitchGDD = bool(int(self.var._configuration.CROP_PARAMETERS['SwitchGDD']))
        # self.var.GDDmethod = int(self.var._configuration.CROP_PARAMETERS['GDDmethod'])
        
        # Declare variables
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

        # self.var.crop_parameters_to_compute = []
        self.var.crop_parameters_to_compute = []
            # 'CC0','SxTop','SxBot','fCO2','tLinSwitch','dHILinear','HIGC',
            # 'CanopyDevEnd','CanopyDevEndCD','Canopy10Pct','Canopy10PctCD','MaxCanopy',
            # 'MaxCanopyCD','HIstartCD','HIend','HIendCD','YldFormCD',
            # 'FloweringEnd','Flowering','FloweringCD','CGC','CDC',
            # 'PlantingDateAdj','HarvestDateAdj',
            # 'CurrentConc']  # TODO: CGC and CDC are in both list - try removing them here?

        # TODO: don't initialize here
        self.var.crop_parameter_names = self.var.crop_parameters_to_read + self.var.crop_parameters_to_compute
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        for param in self.var.crop_parameter_names:
            vars(self.var)[param] = np.copy(arr_zeros)
            
        self.var.fCO2 = arr_zeros.copy()
        self.var.CurrentConc = arr_zeros.copy()
            
        self.read()
        self.compute_crop_parameters()
        
    def read(self):        
        """Function to read crop input parameters"""
        if len(self.var.crop_parameters_to_read) > 0:
            for param in self.var.crop_parameters_to_read:
                read_from_netcdf = file_handling.check_if_nc_has_variable(
                    self.var.CropParameterFileNC,
                    param
                    )
                if read_from_netcdf:
                    d = file_handling.netcdf_to_arrayWithoutTime(
                        self.var.CropParameterFileNC,
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
                        
                    
    def adjust_planting_and_harvesting_date(self):

        if self.var._modelTime.timeStepPCR == 1 or self.var._modelTime.doy == 1:            
            pd = np.copy(self.var.PlantingDate)
            hd = np.copy(self.var.HarvestDate)
            st = self.var._modelTime.currTime
            sd = self.var._modelTime.currTime.timetuple().tm_yday
            isLeapYear1 = calendar.isleap(st.year)            
            pd[(isLeapYear1 & (pd >= 60))] += 1  # TODO: check these
            hd[(isLeapYear1 & (hd >= 60) & (hd > pd))] += 1
            self.var.PlantingDateAdj = np.copy(pd)
            self.var.HarvestDateAdj = np.copy(hd)

    def update_growing_season(self):
        # TODO: check both PlantingDateAdj and HarvestDateAdj are not the same!        
        cond1 = ((self.var.PlantingDateAdj < self.var.HarvestDateAdj)
                 & ((self.var.PlantingDateAdj <= self.var._modelTime.doy)
                    & (self.var._modelTime.doy <= self.var.HarvestDateAdj)))
        cond2 = ((self.var.PlantingDateAdj > self.var.HarvestDateAdj)
                 & ((self.var.PlantingDateAdj <= self.var._modelTime.doy)
                    | (self.var._modelTime.doy <= self.var.HarvestDateAdj)))

        # check the harvest date is before the end of the simulation period
        hd_arr = (np.datetime64(str(datetime.datetime(self.var._modelTime.year, 1, 1))) + np.array(self.var.HarvestDateAdj - 1, dtype='timedelta64[D]'))        
        cond3 = hd_arr <= np.datetime64(str(self.var._modelTime.endTime))

        # check the planting date is after the start of the simulation period
        # N.B. timeStepPCR is the number of time steps since the model started
        cond4 = ((self.var._modelTime.doy - self.var.PlantingDateAdj) < self.var._modelTime.timeStepPCR)

        self.var.GrowingSeason = ((cond1 | cond2) & cond3 & cond4)

        self.var.GrowingSeasonIndex = self.var.GrowingSeason.copy()
        self.var.GrowingSeasonIndex *= np.logical_not(self.var.CropDead | self.var.CropMature)
        self.var.GrowingSeasonDayOne = self.var._modelTime.doy == self.var.PlantingDateAdj
        self.var.DAP[self.var.GrowingSeasonDayOne] = 0

        self.var.GrowingSeasonIndex[self.var.GrowingSeasonDayOne] = True
        self.var.DAP[self.var.GrowingSeasonIndex] += 1
        self.var.DAP[np.logical_not(self.var.GrowingSeasonIndex)] = 0

    def compute_crop_parameters(self):
        """Function to compute additional crop variables required to run 
        AquaCrop"""
        
        # The following adapted from lines 160-240 of AOS_ComputeVariables.m

        # Fractional canopy cover size at emergence
        self.var.CC0 = np.round(10000. * (self.var.PlantPop * self.var.SeedSize) * 10 ** -8) / 10000
        
        # Root extraction terms
        S1 = np.copy(self.var.SxTopQ)
        S2 = np.copy(self.var.SxBotQ)
        self.var.SxTop = np.zeros_like(self.var.SxTopQ)
        self.var.SxBot = np.zeros_like(self.var.SxBotQ)

        cond1 = (S1 == S2)
        self.var.SxTop[cond1] = S1[cond1]
        self.var.SxBot[cond1] = S2[cond1]

        cond2 = np.logical_not(cond1)
        cond21 = (cond2 & (self.var.SxTopQ < self.var.SxBotQ))
        S1[cond21] = self.var.SxBotQ[cond21]
        S2[cond21] = self.var.SxTopQ[cond21]
        xx = 3 * np.divide(S2, (S1 - S2), out=np.zeros_like(S2), where=(S1-S2)!=0)
        SS1 = np.zeros_like(S1)
        SS2 = np.zeros_like(S2)

        cond22 = (cond2 & (xx < 0.5))
        SS1[cond22] = ((4 / 3.5) * S1)[cond22]
        SS2[cond22] = 0

        cond23 = (cond2 & np.logical_not(cond22))
        SS1[cond23] = ((xx + 3.5) * (S1 / (xx + 3)))[cond23]
        SS2[cond23] = (
            (xx - 0.5) *
            np.divide(S2, xx, out=np.zeros_like(S2), where=xx!=0)
        )[cond23]

        cond24 = (cond2 & (self.var.SxTopQ > self.var.SxBotQ))
        self.var.SxTop[cond24] = SS1[cond24]
        self.var.SxBot[cond24] = SS2[cond24]

        cond25 = (cond2 & np.logical_not(cond24))
        self.var.SxTop[cond25] = SS2[cond25]
        self.var.SxBot[cond25] = SS1[cond25]

        # Crop calender
        self.compute_crop_calendar()

        # Harvest index growth coefficient
        self.calculate_HIGC()

        # Days to linear HI switch point
        self.calculate_HI_linear()

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
        
    def calculate_HI_linear(self):
        """Function to calculate time to switch to linear harvest index 
        build-up, and associated linear rate of build-up. Only for 
        fruit/grain crops
        """
        # Determine linear switch point
        cond1 = (self.var.CropType == 3)
        ti = np.zeros_like(self.var.CropType)
        tmax = self.var.YldFormCD
        HIest = np.zeros_like(self.var.HIini)
        HIprev = self.var.HIini

        # Iterate to find linear switch point
        while np.any((self.var.CropType == 3) & (HIest <= self.var.HI0) & (ti < tmax)):    
            ti += 1
            HInew = ((self.var.HIini * self.var.HI0) / (self.var.HIini + (self.var.HI0 - self.var.HIini) * np.exp(-self.var.HIGC * ti)))
            HIest = (HInew + (tmax - ti) * (HInew - HIprev))
            HIprev = HInew

        self.var.tLinSwitch = ti - 1  # Line 19 of AOS_CalculateHILinear.m
        # self.var.tLinSwitch[self.var.CropType != 3] = np.nan
            
        # Determine linear build-up rate
        cond1 = (self.var.tLinSwitch > 0)
        HIest[cond1] = ((self.var.HIini * self.var.HI0) / (self.var.HIini + (self.var.HI0 - self.var.HIini) * np.exp(-self.var.HIGC * self.var.tLinSwitch)))[cond1]
        HIest[np.logical_not(cond1)] = 0
        self.var.dHILinear = ((self.var.HI0 - HIest) / (tmax - self.var.tLinSwitch))  # dHILin will be set to nan in the same cells as tSwitch
                    
    def calculate_HIGC(self):
        """Function to calculate harvest index growth coefficient"""
        # Total yield formation days
        tHI = np.copy(self.var.YldFormCD)
        # print('tHI:',np.min(self.var.YldFormCD),np.max(self.var.YldFormCD))
        cond0 = tHI > 0
        # Iteratively estimate HIGC
        self.var.HIGC = np.full((self.var.HIini.shape), 0.001)
        HIest = np.zeros_like(self.var.HIini)
        while np.any((HIest <= (0.98 * self.var.HI0))[cond0]):
            cond1 = cond0 & (HIest <= (0.98 * self.var.HI0))
            self.var.HIGC[cond1] += 0.001
            HIest = ((self.var.HIini * self.var.HI0) / (self.var.HIini + (self.var.HI0 - self.var.HIini) * np.exp(-self.var.HIGC * tHI)))
        # while np.any(HIest <= (0.98 * self.var.HI0)):
        #     cond1 = (HIest <= (0.98 * self.var.HI0))
        #     self.var.HIGC[cond1] += 0.001
        #     HIest = ((self.var.HIini * self.var.HI0) / (self.var.HIini + (self.var.HI0 - self.var.HIini) * np.exp(-self.var.HIGC * tHI)))

        self.var.HIGC[HIest >= self.var.HI0] -= 0.001

    def compute_crop_calendar(self):
       
        # "Time from sowing to end of vegetative growth period"
        cond1 = (self.var.Determinant == 1)
        self.var.CanopyDevEnd = np.copy(self.var.Senescence)
        self.var.CanopyDevEnd[cond1] = (np.round(self.var.HIstart + (self.var.Flowering / 2)))[cond1]

        # "Time from sowing to 10% canopy cover (non-stressed conditions)
        # self.var.Canopy10Pct = np.round(self.var.Emergence + (np.log(0.1 / self.var.CC0) / self.var.CGC))
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

        # "Time from sowing to maximum canopy cover (non-stressed conditions)
        self.var.MaxCanopy = np.round(self.var.Emergence + (np.log((0.25 * self.var.CCx * self.var.CCx / self.var.CC0) / (self.var.CCx - (0.98 * self.var.CCx))) / self.var.CGC))

        # "Time from sowing to end of yield formation"
        self.var.HIend = self.var.HIstart + self.var.YldForm
        cond2 = (self.var.CropType == 3)

        # TODO: declare these in __init__
        arr_zeros = np.zeros_like(self.var.CropType)
        self.var.FloweringEnd = np.copy(arr_zeros)
        self.var.FloweringEndCD = np.copy(arr_zeros)
        self.var.FloweringCD = np.copy(arr_zeros)
        self.var.FloweringEnd[cond2] = (self.var.HIstart + self.var.Flowering)[cond2]
        self.var.FloweringEndCD[cond2] = self.var.FloweringEnd[cond2]
        self.var.FloweringCD[cond2] = self.var.Flowering[cond2]
        
        # Mode = self.var.CalendarType
        # if Mode == 1:
        if self.var.CalendarType == 1:
            # "Duplicate calendar values (needed to minimise if-statements when
            # switching between GDD and CD runs)
            self.var.EmergenceCD = np.copy(self.var.Emergence)      # only used in this function
            self.var.Canopy10PctCD = np.copy(self.var.Canopy10Pct)  # only used in this function
            self.var.MaxRootingCD = np.copy(self.var.MaxRooting)    # only used in this function
            self.var.SenescenceCD = np.copy(self.var.Senescence)    # only used in this function
            self.var.MaturityCD = np.copy(self.var.Maturity)        # only used in this function
            self.var.MaxCanopyCD = np.copy(self.var.MaxCanopy)
            self.var.CanopyDevEndCD = np.copy(self.var.CanopyDevEnd)
            self.var.HIstartCD = np.copy(self.var.HIstart)
            self.var.HIendCD = np.copy(self.var.HIend)
            self.var.YldFormCD = np.copy(self.var.YldForm)            
            self.var.FloweringEndCD = np.copy(self.var.FloweringEnd)  # only used in this function
            self.var.FloweringCD = np.copy(self.var.Flowering)

        # Pre-compute cumulative GDD during growing season
        if ((self.var.CalendarType == 1) & (self.var.SwitchGDD)) | (self.var.CalendarType == 2):

            pd = np.broadcast_to(
                self.var.PlantingDate,
                (self.var.nFarm, self.var.nCrop, self.var.nCell)
            ).copy()
            hd = np.broadcast_to(
                self.var.HarvestDate,
                (self.var.nFarm, self.var.nCrop, self.var.nCell)
            ).copy()
            sd = self.var._modelTime.startTime.timetuple().tm_yday
            
            # TODO: introduce a check somewhere to ensure that planting date does not equal harvest date
            
            # account for leap years
            isLeapYear1 = calendar.isleap(self.var._modelTime.startTime.year)
            isLeapYear2 = calendar.isleap(self.var._modelTime.startTime.year + 1)
            if isLeapYear1:
                # if harvest day is on or after 29 Feb AND harvest day is
                # greater than planting day, then also adjust harvest day
                # by 1 day
                hd[(hd >= 60) & (hd >= pd)] += 1
                # if planting day is on or after 29 Feb then planting day
                # must be changed by 1 day
                pd[pd >= 60] += 1
            if isLeapYear2:
                # if harvest day is on or after 29 Feb of the second year
                # (=365 + 60) then harvest day must be changed by 1 day
                hd[(hd <= pd) & (hd >= 60)] += 1

            # account for the situation where the harvest date is before the
            # planting date (implying the harvest takes place in the year
            # following that of the planting date). 
            hd[hd < pd] += 365

            # ensure the time period starts after the model start period
            planting_day_before_start_day = sd > pd
            # pd[planting_day_before_start_day] += 365
            # hd[planting_day_before_start_day] += 365
            pd[planting_day_before_start_day] = 0  # This could really screw things up
            hd[planting_day_before_start_day] = 0
                
            max_harvest_date = int(np.max(hd))
            day_idx = np.arange(sd, max_harvest_date + 1)
            day_idx = day_idx[:,None,None,None] * np.ones((self.var.nFarm, self.var.nCrop, self.var.nCell))[None,...]
            growing_season_idx = ((day_idx >= pd) & (day_idx <= hd))
            # print(sd)
            # print(np.shape(day_idx))
            # print(max_harvest_date)
            
            # print(self.var._modelTime.startTime)
            # print(self.var._modelTime.startTime + datetime.timedelta(int(max_harvest_date - sd)))
            tmin = file_handling.netcdf_time_slice_to_array(
                self.var.weather.minDailyTemperatureNC,
                self.var.weather.tminVarName,
                self.var._modelTime.startTime,
                self.var._modelTime.startTime + datetime.timedelta(int(max_harvest_date - sd)),
                cloneMapFileName = self.var.cloneMapFileName,
                LatitudeLongitude = True)

            tmax = file_handling.netcdf_time_slice_to_array(
                self.var.weather.maxDailyTemperatureNC,
                self.var.weather.tmaxVarName,
                self.var._modelTime.startTime,
                self.var._modelTime.startTime + datetime.timedelta(int(max_harvest_date - sd)),
                cloneMapFileName = self.var.cloneMapFileName,
                LatitudeLongitude = True)
            
            # broadcast to crop dimension
            tmin = tmin[...,self.var.landmask]
            tmax = tmax[...,self.var.landmask]
            tmin = tmin[:,None,None,...] * np.ones_like(day_idx)
            tmax = tmax[:,None,None,...] * np.ones_like(day_idx)
            
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

            tmean *= growing_season_idx
            tbase *= growing_season_idx
            GDD = (tmean - tbase)
            GDDcum = np.cumsum(GDD, axis=0)

            # "Check if converting crop calendar to GDD mode"
            if (self.var.CalendarType == 1) & (self.var.SwitchGDD):
                
                # Find GDD equivalent for each crop calendar variable
                m,n,p = pd.shape  # farm,crop,space
                I,J,K = np.ogrid[:m,:n,:p]

                emergence_idx = pd + self.var.EmergenceCD  # crop,lat,lon
                self.var.Emergence = GDDcum[emergence_idx,I,J,K]
                canopy10pct_idx = pd + self.var.Canopy10PctCD
                self.var.Canopy10Pct = GDDcum[canopy10pct_idx,I,J,K]
                maxrooting_idx = pd + self.var.MaxRootingCD
                self.var.MaxRooting = GDDcum[maxrooting_idx,I,J,K]
                maxcanopy_idx = pd + self.var.MaxCanopyCD
                self.var.MaxCanopy = GDDcum[maxcanopy_idx,I,J,K]
                canopydevend_idx = pd + self.var.CanopyDevEndCD
                self.var.CanopyDevEnd = GDDcum[canopydevend_idx,I,J,K]
                senescence_idx = pd + self.var.SenescenceCD
                self.var.Senescence = GDDcum[senescence_idx,I,J,K]
                maturity_idx = pd + self.var.MaturityCD
                self.var.Maturity = GDDcum[maturity_idx,I,J,K]
                histart_idx = pd + self.var.HIstartCD
                self.var.HIstart = GDDcum[histart_idx,I,J,K]
                hiend_idx = pd + self.var.HIendCD
                self.var.HIend = GDDcum[hiend_idx,I,J,K]
                yldform_idx = pd + self.var.YldFormCD
                self.var.YldForm = GDDcum[yldform_idx,I,J,K]

                cond2 = (self.var.CropType == 3)
                floweringend_idx = pd + self.var.FloweringEndCD
                self.var.FloweringEnd[cond2] = GDDcum[floweringend_idx,I,J,K][cond2]
                self.var.Flowering[cond2] = (self.var.FloweringEnd - self.var.HIstart)[cond2]

                # "Convert CGC to GDD mode"
                # self.var.CGC_CD = self.var.CGC
                self.var.CGC = (np.log((((0.98 * self.var.CCx) - self.var.CCx) * self.var.CC0) / (-0.25 * (self.var.CCx ** 2)))) / (-(self.var.MaxCanopy - self.var.Emergence))

                # "Convert CDC to GDD mode"
                # self.var.CDC_CD = self.var.CDC
                tCD = self.var.MaturityCD - self.var.SenescenceCD
                tCD[tCD <= 0] = 1
                tGDD = self.var.Maturity - self.var.Senescence
                tGDD[tGDD <= 0] = 5
                self.var.CDC = (self.var.CCx / tGDD) * np.log(1 + ((1 - self.var.CCi / self.var.CCx) / 0.05))

                # "Set calendar type to GDD mode"
                self.var._configuration.CROP_PARAMETERS['CalendarType'] = "2"
            
            elif self.var.CalendarType == 2:
                
                maxcanopy = np.broadcast_to(self.var.MaxCanopy[None,...], day_idx.shape)
                canopydevend = np.broadcast_to(self.var.CanopyDevEnd[None,...], day_idx.shape)
                histart = np.broadcast_to(self.var.HIstart[None,...], day_idx.shape)
                hiend = np.broadcast_to(self.var.HIend[None,...], day_idx.shape)
                floweringend = np.broadcast_to(self.var.FloweringEnd[None,...], day_idx.shape)
                # "Find calendar days [equivalent] for some variables"

                # "1 Calendar days from sowing to maximum canopy cover"
                
                # TODO: check this indexing
                maxcanopy_idx = np.copy(day_idx)
                maxcanopy_idx[np.logical_not(GDDcum > maxcanopy)] = 999
                maxcanopy_idx = np.nanmin(maxcanopy_idx, axis=0)
                self.var.MaxCanopyCD = maxcanopy_idx - pd + 1

                # "2 Calendar days from sowing to end of vegetative growth"
                canopydevend_idx = np.copy(day_idx)
                canopydevend_idx[np.logical_not(GDDcum > canopydevend)] = 999
                canopydevend_idx = np.nanmin(canopydevend_idx, axis=0)
                self.var.CanopyDevEndCD = canopydevend_idx - pd + 1

                # "3 Calendar days from sowing to start of yield formation"
                histart_idx = np.copy(day_idx)
                histart_idx[np.logical_not(GDDcum > histart)] = 999
                histart_idx = np.nanmin(histart_idx, axis=0)
                self.var.HIstartCD = histart_idx - pd + 1

                # "4 Calendar days from sowing to end of yield formation"
                hiend_idx = np.copy(day_idx)
                hiend_idx[np.logical_not(GDDcum > hiend)] = 999
                hiend_idx = np.nanmin(hiend_idx, axis=0)
                self.var.HIendCD = hiend_idx - pd + 1
                
                # idx = (self.var.HIstartCD == self.var.HIendCD)
                # print(np.argwhere(idx))
                # print(self.var.HIstartCD[idx][0])
                # print(self.var.HIendCD[idx][0])
                # print(self.var.HIstartCD.shape)                
                # print(self.var.HIendCD.shape)
                # print(pd.shape)
                # print(hiend_idx.shape)
                # print(histart_idx.shape)
                # print(pd[idx][0])
                # print(hiend_idx[idx][0])
                # print(histart_idx[idx][0])
                # print(GDDcum[...,50])
                # print(hiend[...,50])
                
                # "Duration of yield formation in calendar days"
                self.var.YldFormCD = self.var.HIendCD - self.var.HIstartCD

                cond1 = (self.var.CropType == 3)
                cond1 = np.broadcast_to(cond1, (self.var.nFarm, self.var.nCrop, self.var.nCell))

                # "1 Calendar days from sowing to end of flowering"
                floweringend_idx = np.copy(day_idx)
                floweringend_idx[np.logical_not(GDDcum > floweringend)] = 999
                floweringend_idx = np.nanmin(floweringend_idx, axis=0)
                FloweringEnd = floweringend_idx - pd + 1

                # "2 Duration of flowering in calendar days"
                # print(cond1.shape)
                # print(self.var.FloweringCD.shape)
                self.var.FloweringCD[cond1] = (FloweringEnd - self.var.HIstartCD)[cond1]
                
    def update_crop_parameters(self):
        """Function to update certain crop parameters for current 
        time step (equivalent to lines 97-163 in 
        compute_crop_calendar)
        """
        pd = np.copy(self.var.PlantingDateAdj)
        hd = np.copy(self.var.HarvestDateAdj)
        hd[hd < pd] += 365
        sd = self.var._modelTime.currTime.timetuple().tm_yday

        # Update certain crop parameters if using GDD mode
        if (self.var.CalendarType == 2):

            cond1 = ((self.var.GrowingSeason) & (self.var._modelTime.doy == pd))
            pd[np.logical_not(cond1)] = 0
            hd[np.logical_not(cond1)] = 0
            max_harvest_date = int(np.max(hd))

            if (max_harvest_date > 0):

                # Dimension (day,crop,cell)
                day_idx = np.arange(sd, max_harvest_date + 1)[:,None,None,None] * np.ones_like(self.var.PlantingDate)[None,:,:,:]
                growing_season_idx = ((day_idx >= pd) & (day_idx <= hd))

                # Extract weather data for first growing season
                tmin = file_handling.netcdf_time_slice_to_array(
                    self.var.weather.minDailyTemperatureNC, self.var.weather.tminVarName,
                    self.var._modelTime.currTime,
                    self.var._modelTime.currTime + datetime.timedelta(int(max_harvest_date - sd)),
                    cloneMapFileName = self.var.cloneMapFileName,
                    LatitudeLongitude = True)

                tmax = file_handling.netcdf_time_slice_to_array(
                    self.var.weather.maxDailyTemperatureNC, self.var.weather.tmaxVarName,
                    self.var._modelTime.currTime,
                    self.var._modelTime.currTime + datetime.timedelta(int(max_harvest_date - sd)),
                    cloneMapFileName = self.var.cloneMapFileName,
                    LatitudeLongitude = True)

                # broadcast to crop dimension
                tmin = tmin[...,self.var.landmask]
                tmax = tmax[...,self.var.landmask]
                tmin = np.broadcast_to(tmin[:,None,None,...], day_idx.shape).copy()
                tmax = np.broadcast_to(tmax[:,None,None,...], day_idx.shape).copy()

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

                tmean[np.logical_not(growing_season_idx)] = 0
                tbase[np.logical_not(growing_season_idx)] = 0
                GDD = (tmean - tbase)
                GDDcum = np.cumsum(GDD, axis=0)
        
                # 1 - Calendar days from sowing to maximum canopy cover
                maxcanopy_idx = np.copy(day_idx)
                maxcanopy_idx[np.logical_not(GDDcum > self.var.MaxCanopy)] = 999
                # maxcanopy_idx[np.logical_not(GDDcum > self.var.MaxCanopy)] = np.nan
                maxcanopy_idx = np.nanmin(maxcanopy_idx, axis=0)
                MaxCanopyCD = (maxcanopy_idx - pd + 1)
                self.var.MaxCanopyCD[cond1] = MaxCanopyCD[cond1]

                # 2 - Calendar days from sowing to end of vegetative growth
                canopydevend_idx = np.copy(day_idx)
                canopydevend_idx[np.logical_not(GDDcum > self.var.CanopyDevEnd)] = 999
                # canopydevend_idx[np.logical_not(GDDcum > self.var.CanopyDevEnd)] = np.nan
                canopydevend_idx = np.nanmin(canopydevend_idx, axis=0)
                CanopyDevEndCD = canopydevend_idx - pd + 1
                self.var.CanopyDevEndCD[cond1] = CanopyDevEndCD[cond1]

                # 3 - Calendar days from sowing to start of yield formation
                histart_idx = np.copy(day_idx)
                histart_idx[np.logical_not(GDDcum > self.var.HIstart)] = 999
                # histart_idx[np.logical_not(GDDcum > self.var.HIstart)] = np.nan
                histart_idx = np.nanmin(histart_idx, axis=0)
                HIstartCD = histart_idx - pd + 1
                self.var.HIstartCD[cond1] = HIstartCD[cond1]

                # 4 - Calendar days from sowing to end of yield formation
                hiend_idx = np.copy(day_idx)
                hiend_idx[np.logical_not(GDDcum > self.var.HIend)] = 999
                # hiend_idx[np.logical_not(GDDcum > self.var.HIend)] = np.nan
                hiend_idx = np.nanmin(hiend_idx, axis=0)
                HIendCD = hiend_idx - pd + 1
                self.var.HIendCD[cond1] = HIendCD[cond1]

                # Duration of yield formation in calendar days
                self.var.YldFormCD[cond1] = (self.var.HIendCD - self.var.HIstartCD)[cond1]

                cond11 = (cond1 & (self.var.CropType == 3))

                # 1 Calendar days from sowing to end of flowering
                floweringend_idx = np.copy(day_idx)
                floweringend_idx[np.logical_not(GDDcum > self.var.FloweringEnd)] = 999
                # floweringend_idx[np.logical_not(GDDcum > self.var.FloweringEnd)] = np.nan
                floweringend_idx = np.nanmin(floweringend_idx, axis=0)
                FloweringEnd = floweringend_idx - pd + 1

                # 2 Duration of flowering in calendar days
                self.var.FloweringCD[cond11] = (FloweringEnd - self.var.HIstartCD)[cond11]

                # Harvest index growth coefficient
                self.calculate_HIGC()

                # Days to linear HI switch point
                self.calculate_HI_linear()

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
                        

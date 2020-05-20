#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import warnings

from .io.CarbonDioxide import refconc

import aquacrop_fc


class CallFortran(object):
    def __init__(self, model):
        self.model = model

    def initial(self):
        # # GrowingDegreeDay
        # self.var.GDDcum = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # self.var.GDD = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # # GrowthStage
        # arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # self.var.GrowthStage = arr_zeros.copy()
        # self.var.time_since_germination = arr_zeros.copy()
        # self.var.time_since_germination_previous = arr_zeros.copy()
        # # CheckGroundwaterTable
        # self.var.th_fc_adj = np.copy(self.var.th_fc_comp)
        # self.var.WTinSoil = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy), dtype=np.int32)
        # # PreIrrigation
        # arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # self.var.PreIrr = np.copy(arr_zeros)
        # self.var.IrrNet = np.copy(arr_zeros)
        # # Drainage
        # self.var.FluxOut = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.domain.nxy))
        # self.var.DeepPerc = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # self.var.Recharge = np.zeros((self.var.domain.nxy))
        # # RainfallPartition
        # arr_zeros = np.zeros((self.var.nFarm,self.var.nCrop, self.var.domain.nxy))        
        # self.var.Runoff = np.copy(arr_zeros)
        # self.var.Infl = np.copy(arr_zeros)
        # # RootZoneWater
        # arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # self.var.thRZ_Act = np.copy(arr_zeros)
        # self.var.thRZ_Sat = np.copy(arr_zeros)
        # self.var.thRZ_Fc = np.copy(arr_zeros)
        # self.var.thRZ_Wp = np.copy(arr_zeros)
        # self.var.thRZ_Dry = np.copy(arr_zeros)
        # self.var.thRZ_Aer = np.copy(arr_zeros)
        # self.var.TAW = np.copy(arr_zeros)
        # self.var.Dr = np.copy(arr_zeros)
        # self.var.Wr = np.copy(arr_zeros)
        # # Irrigation - TODO
        # # Infiltration
        # cond1 = (self.var.Bunds == 0) & (self.var.zBund > 0.001)
        # SurfaceStorage = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # SurfaceStorage[cond1] = self.var.BundWater[cond1]
        # SurfaceStorage = np.clip(SurfaceStorage, None, self.var.zBund)
        # self.var.SurfaceStorage = np.copy(SurfaceStorage)
        # self.var.SurfaceStorageIni = np.copy(SurfaceStorage)
        # # CapillaryRise
        # self.var.CrTot = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # # Germination
        # self.var.DelayedGDDs = np.copy(arr_zeros)
        # self.var.DelayedCDs = np.copy(arr_zeros.astype(np.int32))
        # self.var.Germination = np.copy(arr_zeros.astype(np.int32))
        # # RootDevelopment
        # self.var.rCor = np.ones((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # self.var.Zroot = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # # WaterStress
        # # CanopyCover
        # self.var.tEarlySen = np.copy(arr_zeros)
        # self.var.CC = np.copy(arr_zeros)
        # self.var.CCadj = np.copy(arr_zeros)
        # self.var.CC_NS = np.copy(arr_zeros)
        # self.var.CCadj_NS = np.copy(arr_zeros)
        # self.var.CCxAct = np.copy(arr_zeros)
        # self.var.CCxAct_NS = np.copy(arr_zeros)
        # self.var.CCxW = np.copy(arr_zeros)
        # self.var.CCxW_NS = np.copy(arr_zeros)
        # self.var.CCxEarlySen = np.copy(arr_zeros)
        # self.var.CCprev = np.copy(arr_zeros)
        # self.var.PrematSenes = np.copy(arr_zeros.astype(np.int32))        
        # self.var.CropDead = np.copy(arr_zeros.astype(np.int32))        
        # self.var.CC0adj = np.copy(arr_zeros)
        # # SoilEvaporation
        # arr_zeros = np.zeros(
        #     (self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # self.var.Epot = np.copy(arr_zeros)
        # self.var.Stage2 = np.copy(arr_zeros.astype(bool))
        # self.var.EvapZ = np.copy(arr_zeros)
        # self.var.Wstage2 = np.copy(arr_zeros)
        # self.var.Wsurf = np.copy(arr_zeros)
        # self.var.Wevap_Act = np.copy(arr_zeros)
        # self.var.Wevap_Sat = np.copy(arr_zeros)
        # self.var.Wevap_Fc = np.copy(arr_zeros)
        # self.var.Wevap_Wp = np.copy(arr_zeros)
        # self.var.Wevap_Dry = np.copy(arr_zeros)
        # self.var.dz_xy = np.broadcast_to(self.var.dz[None, None, :, None], (
        #     self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.domain.nxy))
        # self.var.dz_sum_xy = np.broadcast_to(self.var.dz_sum[None, None, :, None], (
        #     self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.domain.nxy))
        # # Transpiration
        # arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # arr_ones = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # self.var.Ksa_Aer = np.copy(arr_zeros)
        # self.var.TrPot0 = np.copy(arr_zeros)
        # self.var.TrPot_NS = np.copy(arr_zeros)
        # self.var.TrAct = np.copy(arr_zeros)
        # self.var.TrAct0 = np.copy(arr_zeros)
        # self.var.AgeDays = np.copy(arr_zeros)
        # self.var.AgeDays_NS = np.copy(arr_zeros)
        # self.var.AerDays = np.copy(arr_zeros)
        # self.var.AerDaysComp  = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.domain.nxy))
        # self.var.Tpot = np.copy(arr_zeros)        
        # self.var.TrRatio = np.copy(arr_ones)
        # self.var.DaySubmerged = np.copy(arr_zeros)
        # # HarvestIndex
        # arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # self.var.YieldForm = np.copy(arr_zeros).astype(np.int32)
        # self.var.HI = np.copy(arr_zeros)
        # self.var.HIt = np.copy(arr_zeros)
        # self.var.PctLagPhase = np.copy(arr_zeros)
        # # BiomassAccumulation
        # self.var.B = np.copy(arr_zeros)
        # self.var.B_NS = np.copy(arr_zeros)
        # # HarvestIndexAdjusted
        # arr_ones = np.ones((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        # self.var.Fpre = np.copy(arr_ones)
        # self.var.Fpost = np.copy(arr_ones)
        # self.var.fpost_dwn = np.copy(arr_ones)
        # self.var.fpost_upp = np.copy(arr_ones)
        # self.var.Fpol = np.copy(arr_zeros)
        # self.var.sCor1 = np.copy(arr_zeros)
        # self.var.sCor2 = np.copy(arr_zeros)
        # self.var.HIadj = np.copy(arr_zeros)
        # self.var.PreAdj = np.copy(arr_zeros).astype(np.int32)
        
        pass

    def adjust_root_zone_depletion(self):
        rootdepth = np.maximum(self.model.Zmin, self.model.Zroot)
        AbvFc = ((self.model.thRZ_Act - self.model.thRZ_Fc) * 1000 * rootdepth)
        AbvFc = np.clip(AbvFc, 0., None)
        WCadj = self.model.ETpot - self.model.weather.precipitation + self.model.Runoff - AbvFc
        Dr = self.model.Dr + WCadj
        Dr = np.clip(Dr, 0., None)
        return Dr
    
    def compute_irrigation_depth_soil_moisture_threshold(self, WCadj):
        # If irrigation is based on soil moisture, get the soil moisture
        # target for the current growth stage and determine threshold to
        # initiate irrigation
        I,J,K = np.ogrid[:self.model.nFarm,:self.model.nCrop,:self.model.domain.nxy]
        growth_stage_index = self.model.GrowthStage.astype(int) - 1
        SMT = np.concatenate((self.model.SMT1[None,:],
                              self.model.SMT2[None,:],
                              self.model.SMT3[None,:],
                              self.model.SMT4[None,:]), axis=0)
        SMT = SMT[growth_stage_index,I,J,K]
        IrrThr = np.round(((1. - SMT / 100.) * self.model.TAW), 3)
        IrrReq = self.adjust_root_zone_depletion()
        EffAdj = ((100. - self.model.AppEff) + 100.) / 100.  # ???
        IrrReq *= EffAdj
        irrigate = self.model.GrowingSeasonIndex & self.model.irrigate_soil_moisture_threshold & (Dr > IrrThr)
        self.model.Irr[irrigate] = np.clip(IrrReq, 0, self.model.MaxIrr)[irrigate]

    def compute_irrigation_depth_fixed_interval(self):
        """Irrigation depth using fixed interval method."""
        IrrReq = self.adjust_root_zone_depletion()
        EffAdj = ((100. - self.model.AppEff) + 100.) / 100.
        IrrReq *= EffAdj
        IrrReq = np.clip(IrrReq, 0., self.model.MaxIrr)
        nDays = self.model.DAP - 1
        irrigate = self.model.GrowingSeasonIndex & self.model.irrigate_fixed_interval & ((nDays % self.model.IrrInterval) == 0)
        self.model.Irr[irrigate] = IrrReq[irrigate]

    def compute_irrigation_depth_schedule(self):
        # If irrigation is based on a pre-defined schedule then the irrigation
        # requirement for each crop is read from a netCDF file. Note that if
        # the option 'irrScheduleFileNC' is None, then nothing will be imported
        # and the irrigation requirement will be zero
        IrrReq = np.zeros((self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        if self.model.irrScheduleFileNC != None:            
            IrrReq = file_handling.netcdf_to_array(
                self.model.irrScheduleFileNC,
                "irrigation_depth",
                str(self.model._modelTime.fulldate), 
                cloneMapFileName = self.model.cloneMapFileName
            )
            IrrReq = IrrReq[self.model.landmask_crop].reshape(self.model.nCrop,self.model.domain.nxy)
            
        irrigate = self.model.GrowingSeasonIndex & self.model.irrigate_from_schedule
        self.model.Irr[irrigate] = IrrReq[irrigate]

    def compute_irrigation_depth_net(self):        
        # Note that if irrigation is based on net irrigation then it is
        # performed after calculation of transpiration. A dummy method is
        # included here in order to provide this explanation.
        pass
    
    def compute_irrigation_depth(self):
        self.model.Irr[:] = 0.
        if np.any(self.model.GrowingSeasonIndex):
            if np.any(self.model.irrigate_soil_moisture_threshold):
                self.compute_irrigation_depth_soil_moisture_threshold()
            if np.any(self.model.irrigate_fixed_interval):
                self.compute_irrigation_depth_fixed_interval()                    
            if np.any(self.model.irrigate_from_schedule):
                self.compute_irrigation_depth_schedule()
            if np.any(self.model.irrigate_net):
                self.compute_irrigation_depth_net()
        self.model.IrrCum += self.model.Irr
        self.model.IrrCum[np.logical_not(self.model.GrowingSeasonIndex)] = 0
    
    def dynamic_irrigation(self):
        """Function to get irrigation depth for the current day"""
        if np.any(self.model.GrowingSeasonDayOne):
            self.model.IrrCum[self.model.GrowingSeasonDayOne] = 0
            self.model.IrrNetCum[self.model.GrowingSeasonDayOne] = 0
        self.compute_irrigation_depth()

    def dynamic_water_stress(self, beta):
        """Function to calculate water stress coefficients"""
        p_up = np.concatenate(
            (self.model.p_up1[None, ...],
             self.model.p_up2[None, ...],
             self.model.p_up3[None, ...],
             self.model.p_up4[None, ...]), axis=0)

        p_lo = np.concatenate(
            (self.model.p_lo1[None, ...],
             self.model.p_lo2[None, ...],
             self.model.p_lo3[None, ...],
             self.model.p_lo4[None, ...]), axis=0)

        fshape_w = np.concatenate(
            (self.model.fshape_w1[None, ...],
             self.model.fshape_w2[None, ...],
             self.model.fshape_w3[None, ...],
             self.model.fshape_w4[None, ...]), axis=0)

        # et0 = np.broadcast_to(self.model.referencePotET[None,None,:], (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        et0 = self.model.model.etref.values.copy()
        # et0 = self.model.weather.referencePotET.copy()
        # et0 = (self.model.referencePotET[None,:] * np.ones((self.model.nCrop))[:,None])

        # Adjust stress thresholds for Et0 on current day (don't do this for
        # pollination water stress coefficient)
        cond1 = (self.model.ETadj == 1.)
        for stress in range(3):
            p_up[stress, ...][cond1] = (
                p_up[stress, ...] + (0.04 * (5. - et0)) * (np.log10(10. - 9. * p_up[stress, ...])))[cond1]
            p_lo[stress, ...][cond1] = (
                p_lo[stress, ...] + (0.04 * (5. - et0)) * (np.log10(10. - 9. * p_lo[stress, ...])))[cond1]

        # Adjust senescence threshold if early senescence triggered
        if beta:
            cond2 = (self.model.tEarlySen > 0.)
            p_up[2, ...][cond2] = (
                p_up[2, ...] * (1. - (self.model.beta / 100.)))[cond2]

        # Limit adjusted values
        p_up = np.clip(p_up, 0, 1)
        p_lo = np.clip(p_lo, 0, 1)

        # Calculate relative depletion
        Drel = np.zeros(
            (4, self.model.nFarm, self.model.nCrop, self.model.domain.nxy))

        # 1 - No water stress
        cond1 = (self.model.Dr <= (p_up * self.model.TAW))
        # print(self.model.Dr)
        # print(p_up)
        # print('p_up:', p_up)
        Drel[cond1] = 0

        # 2 - Partial water stress
        cond2 = (self.model.Dr > (p_up * self.model.TAW)) & (self.model.Dr <
                                                         (p_lo * self.model.TAW)) & np.logical_not(cond1)
        x1 = p_lo - np.divide(self.model.Dr, self.model.TAW,
                              out=np.zeros_like(Drel), where=self.model.TAW != 0)
        x2 = p_lo - p_up
        Drel[cond2] = (
            1. - np.divide(x1, x2, out=np.zeros_like(Drel), where=x2 != 0))[cond2]

        # 3 - Full water stress
        cond3 = (self.model.Dr >= (p_lo * self.model.TAW)
                 ) & np.logical_not(cond1 | cond2)
        Drel[cond3] = 1.

        # Calculate root zone stress coefficients
        idx = np.arange(0, 3)
        x1 = np.exp(Drel[idx, ...] * fshape_w[idx, ...]) - 1.
        x2 = np.exp(fshape_w[idx, ...]) - 1.
        Ks = (1. - np.divide(x1, x2, out=np.zeros_like(x2), where=x2 != 0))

        # Water stress coefficients (leaf expansion, stomatal closure,
        # senescence, pollination failure)
        self.model.Ksw_Exp = np.copy(Ks[0, ...])
        self.model.Ksw_Sto = np.copy(Ks[1, ...])
        self.model.Ksw_Sen = np.copy(Ks[2, ...])
        self.model.Ksw_Pol = 1. - Drel[3, ...]

        # Mean water stress coefficient for stomatal closure
        self.model.Ksw_StoLin = 1. - Drel[1, ...]


    def temperature_stress_biomass(self):
        """Function to calculate temperature stress coefficient 
        affecting biomass growth
        """
        KsBio_up = 1
        KsBio_lo = 0.02
        fshapeb = -1 * (np.log(((KsBio_lo * KsBio_up) - 0.98 * KsBio_lo) / (0.98 * (KsBio_up - KsBio_lo))))
        cond1 = (self.model.BioTempStress == 0)
        self.model.Kst_Bio[cond1] = 1
        cond2 = (self.model.BioTempStress == 1)
        cond21 = (cond2 & (self.model.GDD >= self.model.GDD_up))
        self.model.Kst_Bio[cond21] = 1
        cond22 = (cond2 & (self.model.GDD <= self.model.GDD_lo))
        self.model.Kst_Bio[cond22] = 0
        cond23 = (cond2 & np.logical_not(cond21 | cond22))
        GDDrel_divd = (self.model.GDD - self.model.GDD_lo)
        GDDrel_divs = (self.model.GDD_up - self.model.GDD_lo)
        GDDrel = np.divide(GDDrel_divd, GDDrel_divs, out=np.zeros_like(GDDrel_divs), where=GDDrel_divs!=0)
        Kst_Bio_divd = (KsBio_up * KsBio_lo)
        Kst_Bio_divs = (KsBio_lo + (KsBio_up - KsBio_lo) * np.exp(-fshapeb * GDDrel))        
        self.model.Kst_Bio[cond23] = np.divide(Kst_Bio_divd, Kst_Bio_divs, out=np.zeros_like(Kst_Bio_divs), where=Kst_Bio_divs!=0)[cond23]
        self.model.Kst_Bio[cond23] = (self.model.Kst_Bio - KsBio_lo * (1 - GDDrel))[cond23]
        
    def temperature_stress_heat(self, KsPol_up, KsPol_lo):
        """Function to calculate effects of heat stress on 
        pollination
        """
        cond3 = (self.model.PolHeatStress == 0)
        self.model.Kst_PolH[cond3] = 1
        cond4 = (self.model.PolHeatStress == 1)
        cond41 = (
            cond4
            & (self.model.model.tmax.values <= self.model.Tmax_lo)
            # & (self.model.weather.tmax <= self.model.Tmax_lo)
        )
        self.model.Kst_PolH[cond41] = 1
        cond42 = (
            cond4
            & (self.model.model.tmax.values >= self.model.Tmax_up)
            # & (self.model.weather.tmax >= self.model.Tmax_up)
        )
        self.model.Kst_PolH[cond42] = 0
        cond43 = (cond4 & np.logical_not(cond41 | cond42))
        Trel_divd = (self.model.model.tmax.values - self.model.Tmax_lo)
        # Trel_divd = (self.model.weather.tmax - self.model.Tmax_lo)
        Trel_divs = (self.model.Tmax_up - self.model.Tmax_lo)
        Trel = np.divide(
            Trel_divd,
            Trel_divs,
            out=np.zeros_like(Trel_divs),
            where=Trel_divs!=0
        )
        Kst_PolH_divd = (KsPol_up * KsPol_lo)
        Kst_PolH_divs = (
            KsPol_lo
            + (KsPol_up - KsPol_lo)
            * np.exp(-self.model.fshape_b * (1. - Trel))
        )
        self.model.Kst_PolH[cond43] = np.divide(
            Kst_PolH_divd,
            Kst_PolH_divs,
            out=np.zeros_like(Kst_PolH_divs),
            where=Kst_PolH_divs!=0
        )[cond43]
        
    def temperature_stress_cold(self, KsPol_up, KsPol_lo):
        """Function to calculate effects of cold stress on 
        pollination
        """
        # tmin = np.broadcast_to(self.model.tmin[None,None,:], (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        tmin = self.model.model.tmin.values.copy()
        # tmin = self.model.weather.tmin.copy()
        # tmin = self.model.tmin[None,:] * np.ones((self.model.nCrop))[:,None]
        cond5 = (self.model.PolColdStress == 0)
        self.model.Kst_PolC[cond5] = 1
        cond6 = (self.model.PolColdStress == 1)
        cond61 = (cond6 & (tmin >= self.model.Tmin_up))
        self.model.Kst_PolC[cond61] = 1
        cond62 = (cond6 & (tmin <= self.model.Tmin_lo))
        self.model.Kst_PolC[cond62] = 0
        Trel_divd = (self.model.Tmin_up - tmin)
        Trel_divs = (self.model.Tmin_up - self.model.Tmin_lo)
        Trel = np.divide(Trel_divd, Trel_divs, out=np.zeros_like(Trel_divs), where=Trel_divs!=0)
        Kst_PolC_divd = (KsPol_up * KsPol_lo)
        Kst_PolC_divs = (KsPol_lo + (KsPol_up - KsPol_lo) * np.exp(-self.model.fshape_b * (1 - Trel)))
        self.model.Kst_PolC[cond62] = np.divide(Kst_PolC_divd, Kst_PolC_divs, out=np.zeros_like(Kst_PolC_divs), where=Kst_PolC_divs!=0)[cond62]
        
    def dynamic_temperature_stress(self):
        """Function to calculate temperature stress coefficients"""
        self.temperature_stress_biomass()
        KsPol_up = 1
        KsPol_lo = 0.001
        self.temperature_stress_heat(KsPol_up, KsPol_lo)
        self.temperature_stress_cold(KsPol_up, KsPol_lo)

    def dynamic_yield(self):
        """Function to calculate crop yield"""
        if np.any(self.model.GrowingSeasonDayOne):
            self.model.CropMature[self.model.GrowingSeasonDayOne] = False
        self.model.Y[self.model.GrowingSeasonIndex] = (
            (self.model.B / 100) * self.model.HIadj
        )[self.model.GrowingSeasonIndex]
        is_mature_calendar_type_one = (
            (self.model.CalendarType == 1)
            & ((self.model.DAP - self.model.DelayedCDs) >= self.model.Maturity)
        )
        is_mature_calendar_type_two = (
            (self.model.CalendarType == 2)
            & ((self.model.GDDcum - self.model.DelayedGDDs) >= self.model.Maturity)
        )
        is_mature = (
            self.model.GrowingSeasonIndex
            & (is_mature_calendar_type_one | is_mature_calendar_type_two)
        )
        self.model.CropMature[is_mature] = True
        self.model.Y[np.logical_not(self.model.GrowingSeasonIndex)] = 0
        
    def dynamic(self):

        # ############################# #
        # GrowingDegreeDay
        # ############################# #
        aquacrop_fc.gdd_w.update_gdd_w(
            self.model.GDD.T, 
            self.model.GDDcum.T, 
            self.model.GDDmethod, 
            self.model.model.tmax.values.T, 
            self.model.model.tmin.values.T,
            self.model.Tbase.T,
            self.model.Tupp.T,
            self.model.GrowingSeasonIndex.T, 
            self.model.nFarm, self.model.nCrop, self.model.domain.nxy
            )
        
        # ############################# #
        # GrowthStage
        # ############################# #
        aquacrop_fc.growth_stage_w.update_growth_stage_w(
            np.int32(self.model.GrowthStage).T,
            self.model.Canopy10Pct.T,
            self.model.MaxCanopy.T,
            self.model.Senescence.T,
            self.model.GDDcum.T,
            np.int32(self.model.DAP).T,
            self.model.DelayedCDs.T,
            self.model.DelayedGDDs.T,
            int(self.model.CalendarType),
            self.model.GrowingSeasonIndex.T,
            self.model.nFarm,
            self.model.nCrop,
            self.model.domain.nxy
        )        

        # ############################# #
        # Initial condition
        # ############################# #

        # Condition to identify crops which are not being grown or crops which
        # have only just finished being grown. The water content of crops
        # meeting this condition is used to compute the area-weighted initial
        # condition
        if np.any(self.model.GrowingSeasonDayOne):
            cond1 = np.logical_not(self.model.GrowingSeasonIndex) | self.model.GrowingSeasonDayOne
            cond1 = np.broadcast_to(cond1[:,:,None,:], self.model.th.shape)
            th = np.copy(self.model.th)
            th[(np.logical_not(cond1))] = np.nan

            # TEMPORARY FIX
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                th_ave = np.nanmean(th, axis=0) # average along farm dimension

            th_ave = np.broadcast_to(
                th_ave,
                (self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.domain.nxy)
            )
            cond2 = np.broadcast_to(self.model.GrowingSeasonDayOne[:,:,None,:], self.model.th.shape)
            self.model.th[cond2] = th_ave[cond2]

        # ############################# #
        # CheckGroundwaterTable
        # ############################# #
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.check_gw_table_w.update_check_gw_table_w(
            self.model.th.T,
            self.model.th_fc_adj.T,
            np.int32(self.model.WTinSoil).T,
            self.model.th_sat.T,
            self.model.th_fc.T,
            int(self.model.groundwater.WaterTable),
            int(self.model.groundwater.DynamicWaterTable),
            self.model.groundwater.zGW,
            self.model.dz,
            layer_ix,
            self.model.nFarm,
            self.model.nCrop,
            self.model.nComp,
            self.model.nLayer,
            self.model.domain.nxy
        )

        # ############################# #
        # PreIrrigation
        # ############################# #        
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.pre_irr_w.update_pre_irr_w(
            self.model.PreIrr.T,
            self.model.th.T,
            self.model.IrrMethod.T,
            self.model.DAP.T,
            self.model.Zroot.T,
            self.model.Zmin.T,
            self.model.NetIrrSMT.T,
            self.model.th_fc.T,
            self.model.th_wilt.T,
            self.model.dz,
            self.model.dz_sum,
            layer_ix,
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
            )

        # ############################# #        
        # Drainage
        # ############################# #        
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.drainage_w.update_drainage_w(
            self.model.th.T,
            self.model.DeepPerc.T,
            self.model.FluxOut.T,
            self.model.th_sat.T,
            self.model.th_fc.T,
            self.model.k_sat.T,
            self.model.tau.T,
            self.model.th_fc_adj.T,
            self.model.dz,
            self.model.dz_sum,
            layer_ix,
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
            )

        # ############################# #
        # RainfallPartition
        # ############################# #
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.rainfall_partition_w.update_rain_part_w(
            self.model.Runoff.T,
            self.model.Infl.T,
            self.model.model.prec.values.T,
            # self.model.weather.precipitation.T,
            self.model.th.T,
            np.int32(self.model.DaySubmerged).T,
            np.int32(self.model.Bunds).T,
            self.model.zBund.T,
            self.model.th_fc.T,
            self.model.th_wilt.T,
            np.int32(self.model.CN).T,
            np.int32(self.model.adjustCurveNumber),
            self.model.zCN.T,
            self.model.CNbot.T,
            self.model.CNtop.T,
            self.model.dz,
            self.model.dz_sum,
            layer_ix,
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
            )

        # ############################# #
        # RootZoneWater
        # ############################# #
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.root_zone_water_w.update_root_zone_water_w(
            self.model.thRZ_Act.T, 
            self.model.thRZ_Sat.T, 
            self.model.thRZ_Fc.T, 
            self.model.thRZ_Wp.T, 
            self.model.thRZ_Dry.T, 
            self.model.thRZ_Aer.T, 
            self.model.TAW.T, 
            self.model.Dr.T, 
            self.model.th.T, 
            self.model.th_sat.T, 
            self.model.th_fc.T, 
            self.model.th_wilt.T, 
            self.model.th_dry.T, 
            self.model.Aer.T, 
            self.model.Zroot.T, 
            self.model.Zmin.T, 
            self.model.dz, 
            self.model.dz_sum, 
            layer_ix, 
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        )

        # ############################# #
        # Irrigation
        # ############################# #

        # TODO
        self.dynamic_irrigation()
        
        # ############################# #
        # Infiltration
        # ############################# #
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.infiltration_w.update_infl_w(
            self.model.Infl.T,
            self.model.SurfaceStorage.T,
            self.model.FluxOut.T,
            self.model.DeepPerc.T,
            self.model.Runoff.T,
            self.model.th.T,
            self.model.Irr.T,
            self.model.AppEff.T,
            self.model.Bunds.T,
            self.model.zBund.T,
            self.model.th_sat.T,
            self.model.th_fc.T,
            self.model.th_fc_adj.T,
            self.model.k_sat.T,
            self.model.tau.T,
            self.model.dz,
            layer_ix,
            self.model.nFarm,
            self.model.nCrop,
            self.model.nComp,
            self.model.nLayer,
            self.model.domain.nxy
            )
        

        # ############################# #
        # CapillaryRise
        # ############################# #
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.capillary_rise_w.update_cap_rise_w(
            self.model.CrTot.T,
            self.model.th.T,
            self.model.th_wilt.T,
            self.model.th_fc.T,
            self.model.th_fc_adj.T,
            self.model.k_sat.T,
            self.model.aCR.T,
            self.model.bCR.T,
            self.model.fshape_cr.T,
            self.model.FluxOut.T,
            self.model.groundwater.WaterTable,
            self.model.groundwater.zGW,
            self.model.dz,
            self.model.dz_layer,
            layer_ix,
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
            )

        # ############################# #
        # Germination
        # ############################# #
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.germination_w.update_germ_w(
            self.model.Germination.T,
            self.model.DelayedCDs.T,
            self.model.DelayedGDDs.T,
            self.model.GDD.T,
            self.model.th.T,
            self.model.th_fc.T,
            self.model.th_wilt.T,
            self.model.zGerm.T,
            self.model.GermThr.T,
            self.model.dz,
            self.model.dz_sum,
            layer_ix,
            self.model.GrowingSeasonIndex.T,
            self.model.nFarm,
            self.model.nCrop,
            self.model.nComp,
            self.model.nLayer,
            self.model.domain.nxy
        )

        # ############################# #
        # RootDevelopment
        # ############################# #

        # TODO - need to add reset_initial_conditions() to Fortran module
        if np.any(self.model.GrowingSeasonDayOne):
            self.model.rCor[self.model.GrowingSeasonDayOne] = 1
            self.model.Zroot[self.model.GrowingSeasonDayOne] = self.model.Zmin[self.model.GrowingSeasonDayOne]

        aquacrop_fc.root_dev_w.update_root_dev_w(
            self.model.Zroot.T, 
            self.model.rCor.T, 
            self.model.Zmin.T, 
            self.model.Zmax.T, 
            self.model.PctZmin.T, 
            self.model.Emergence.T, 
            self.model.MaxRooting.T, 
            self.model.fshape_r.T, 
            self.model.fshape_ex.T, 
            self.model.SxBot.T,
            self.model.SxTop.T,
            self.model.DAP.T,
            self.model.GDD.T,
            self.model.GDDcum.T,
            self.model.DelayedCDs.T,
            self.model.DelayedGDDs.T,
            self.model.TrRatio.T,
            self.model.Germination.T, 
            self.model.zRes.T,
            self.model.groundwater.WaterTable, 
            self.model.groundwater.zGW, 
            self.model.CalendarType, 
            self.model.GrowingSeasonIndex.T,
            self.model.nFarm, self.model.nCrop, self.model.domain.nxy
        )
        
        # ############################# #
        # RootZoneWater
        # ############################# #
        
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.root_zone_water_w.update_root_zone_water_w(
            self.model.thRZ_Act.T, 
            self.model.thRZ_Sat.T, 
            self.model.thRZ_Fc.T, 
            self.model.thRZ_Wp.T, 
            self.model.thRZ_Dry.T, 
            self.model.thRZ_Aer.T, 
            self.model.TAW.T, 
            self.model.Dr.T, 
            self.model.th.T, 
            self.model.th_sat.T, 
            self.model.th_fc.T, 
            self.model.th_wilt.T, 
            self.model.th_dry.T, 
            self.model.Aer.T, 
            self.model.Zroot.T, 
            self.model.Zmin.T, 
            self.model.dz, 
            self.model.dz_sum, 
            layer_ix, 
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        )
        
        # ############################# #
        # WaterStress
        # ############################# #

        # TODO - convert to Fortran
        self.dynamic_water_stress(beta=True)

        # ############################# #        
        # CanopyCover
        # ############################# #
        aquacrop_fc.canopy_cover_w.update_canopy_cover_w(
            self.model.CC.T,
            self.model.CCprev.T,
            self.model.CCadj.T,
            self.model.CC_NS.T,
            self.model.CCadj_NS.T,
            self.model.CCxW.T,
            self.model.CCxAct.T,
            self.model.CCxW_NS.T,
            self.model.CCxAct_NS.T,
            self.model.CC0adj.T,
            self.model.CCxEarlySen.T,
            self.model.tEarlySen.T,
            np.int32(self.model.PrematSenes).T,  # not required when all modules use Fortran
            self.model.CropDead.T,
            self.model.GDD.T,
            self.model.GDDcum.T,
            self.model.CC0.T,
            self.model.CCx.T,
            self.model.CGC.T,
            self.model.CDC.T,
            self.model.Emergence.T,
            self.model.Maturity.T,
            self.model.Senescence.T,
            self.model.CanopyDevEnd.T,
            self.model.Dr.T,
            self.model.TAW.T,
            self.model.model.etref.values.T,
            self.model.ETadj.T,
            self.model.p_up1.T,
            self.model.p_up2.T,
            self.model.p_up3.T,
            self.model.p_up4.T,
            self.model.p_lo1.T,
            self.model.p_lo2.T,
            self.model.p_lo3.T,
            self.model.p_lo4.T,
            self.model.fshape_w1.T,
            self.model.fshape_w2.T,
            self.model.fshape_w3.T,
            self.model.fshape_w4.T,
            self.model.GrowingSeasonIndex.T,
            self.model.GrowingSeasonDayOne.T,
            int(self.model.CalendarType),
            self.model.DAP.T,
            self.model.DelayedCDs.T,
            self.model.DelayedGDDs.T,
            int(self.model.nFarm),
            int(self.model.nCrop),
            int(self.model.domain.nxy)
        )

        # ############################# #
        # SoilEvaporation
        # ############################# #
        self.model.EsAct = np.zeros(
            (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        prec = np.broadcast_to(self.model.model.prec.values,
                               (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        etref = np.broadcast_to(self.model.model.etref.values,
                                (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        EvapTimeSteps = 20
        aquacrop_fc.soil_evaporation_w.update_soil_evap_w(
            np.float64(prec).T,
            np.float64(etref).T,
            self.model.EsAct.T,
            self.model.Epot.T,
            self.model.Irr.T,
            np.int32(self.model.IrrMethod).T,
            self.model.Infl.T,
            self.model.th.T,
            self.model.th_sat_comp.T,
            self.model.th_fc_comp.T,
            self.model.th_wilt_comp.T,
            self.model.th_dry_comp.T,
            self.model.SurfaceStorage.T,
            self.model.WetSurf.T,
            self.model.Wsurf.T,
            self.model.Wstage2.T,
            self.model.CC.T,
            self.model.CCadj.T,
            self.model.CCxAct.T,
            self.model.EvapZ.T,
            self.model.EvapZmin.T,
            self.model.EvapZmax.T,
            self.model.REW.T,
            self.model.Kex.T,
            self.model.CCxW.T,
            self.model.fwcc.T,
            self.model.fevap.T,
            self.model.fWrelExp.T,
            self.model.dz.T,
            self.model.dz_sum.T,
            np.int32(self.model.Mulches).T,
            self.model.fMulch.T,
            self.model.MulchPctGS.T,
            self.model.MulchPctOS.T,
            np.int32(self.model.GrowingSeasonIndex).T,
            np.int32(self.model.Senescence).T,
            np.int32(self.model.PrematSenes).T,
            np.int32(self.model.CalendarType),
            np.int32(self.model.DAP).T,
            np.int32(self.model.GDDcum).T,
            np.int32(self.model.DelayedCDs).T,
            np.int32(self.model.DelayedGDDs).T,
            self.model.model.time.timestep,
            EvapTimeSteps,
            self.model.nFarm,
            self.model.nCrop,
            self.model.nComp,
            self.model.domain.nxy
        )

        # ############################# #
        # RootZoneWater
        # ############################# #
        
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.root_zone_water_w.update_root_zone_water_w(
            self.model.thRZ_Act.T, 
            self.model.thRZ_Sat.T, 
            self.model.thRZ_Fc.T, 
            self.model.thRZ_Wp.T, 
            self.model.thRZ_Dry.T, 
            self.model.thRZ_Aer.T, 
            self.model.TAW.T, 
            self.model.Dr.T, 
            self.model.th.T, 
            self.model.th_sat.T, 
            self.model.th_fc.T, 
            self.model.th_wilt.T, 
            self.model.th_dry.T, 
            self.model.Aer.T, 
            self.model.Zroot.T, 
            self.model.Zmin.T, 
            self.model.dz, 
            self.model.dz_sum, 
            layer_ix, 
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        )
        
        # ############################# #
        # WaterStress
        # ############################# #

        # TODO - convert to Fortran
        self.dynamic_water_stress(beta=True)

        # ############################# #
        # Transpiration
        # ############################# #
        
        # reset initial conditions
        if np.any(self.model.GrowingSeasonDayOne):
            self.model.AgeDays[self.model.GrowingSeasonDayOne] = 0  # not sure if required
            self.model.AgeDays_NS[self.model.GrowingSeasonDayOne] = 0  # not sure if required
            cond = self.model.GrowingSeasonDayOne
            cond_comp = np.broadcast_to(cond[:,:,None,:], self.model.AerDaysComp.shape)
            self.model.AerDays[cond] = 0
            self.model.AerDaysComp[cond_comp] = 0        
            self.model.Tpot[cond] = 0
            self.model.TrRatio[cond] = 1
            # self.model.TrAct[cond] = 0  # TEMP - may not require?
            self.model.DaySubmerged[cond] = 0
            # self.reset_initial_conditions()

        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.transpiration_w.update_transpiration_w(
            self.model.TrPot0.T, 
            self.model.TrPot_NS.T, 
            self.model.TrAct.T,
            self.model.TrAct0.T, 
            self.model.Tpot.T, 
            self.model.TrRatio.T,
            np.int32(self.model.AerDays).T, 
            np.int32(self.model.AerDaysComp).T, 
            self.model.th.T, 
            self.model.thRZ_Act.T, 
            self.model.thRZ_Sat.T, 
            self.model.thRZ_Fc.T,
            self.model.thRZ_Wp.T, 
            self.model.thRZ_Dry.T, 
            self.model.thRZ_Aer.T, 
            self.model.TAW.T, 
            self.model.Dr.T,
            np.int32(self.model.AgeDays).T,
            np.int32(self.model.AgeDays_NS).T,
            np.int32(self.model.DaySubmerged).T,
            self.model.SurfaceStorage.T, 
            self.model.IrrNet.T, 
            self.model.IrrNetCum.T, 
            self.model.CC.T, 
            self.model.model.etref.values.T, 
            # self.model.weather.referencePotET.T, 
            self.model.th_sat.T, 
            self.model.th_fc.T, 
            self.model.th_wilt.T, 
            self.model.th_dry.T, 
            np.int32(self.model.MaxCanopyCD).T, 
            self.model.Kcb.T, 
            self.model.Ksw_StoLin.T, 
            self.model.CCadj.T, 
            self.model.CCadj_NS.T,
            self.model.CCprev.T, 
            self.model.CCxW.T,
            self.model.CCxW_NS.T,
            self.model.Zroot.T,
            self.model.rCor.T,
            self.model.Zmin.T,
            self.model.a_Tr.T,
            self.model.Aer.T,
            self.model.fage.T, 
            np.int32(self.model.LagAer).T, 
            self.model.SxBot.T, 
            self.model.SxTop.T, 
            np.int32(self.model.ETadj).T,
            self.model.p_lo2.T, 
            self.model.p_up2.T, 
            self.model.fshape_w2.T, 
            np.int32(self.model.IrrMethod).T,
            self.model.NetIrrSMT.T,
            self.model.CurrentConc.T, 
            refconc,
            # refconc,
            np.int32(self.model.DAP).T,
            np.int32(self.model.DelayedCDs).T,
            self.model.dz.T, 
            self.model.dz_sum.T, 
            np.int32(layer_ix).T, 
            np.int32(self.model.GrowingSeasonIndex).T,
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
            )

        # ############################# #
        # Evapotranspiration
        # ############################# #

        # TODO: add this to the end of transpiration module
        self.model.ETpot = self.model.Epot + self.model.Tpot

        # ############################# #
        # Inflow
        # ############################# #
        self.model.GwIn = np.zeros((self.model.nCrop, self.model.nFarm, self.model.domain.nxy))
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.inflow_w.update_inflow_w(
            self.model.GwIn.T,
            self.model.th.T,
            np.int32(self.model.groundwater.WaterTable),
            self.model.groundwater.zGW.T,
            self.model.th_sat.T,
            self.model.dz,
            layer_ix,
            self.model.nFarm,
            self.model.nCrop,
            self.model.nComp,
            self.model.nLayer,
            self.model.domain.nxy
            )

        # ############################# #
        # HarvestIndex
        # ############################# #
        aquacrop_fc.harvest_index_w.update_harvest_index_w(
            self.model.HI.T, 
            self.model.PctLagPhase.T,
            self.model.YieldForm.T,
            self.model.CCprev.T, 
            self.model.CCmin.T, 
            self.model.CCx.T, 
            self.model.HIini.T, 
            self.model.HI0.T, 
            self.model.HIGC.T, 
            self.model.HIstart.T, 
            self.model.HIstartCD.T, 
            self.model.tLinSwitch.T, 
            self.model.dHILinear.T, 
            self.model.GDDcum.T, 
            self.model.DAP.T, 
            self.model.DelayedCDs.T, 
            self.model.DelayedGDDs.T, 
            self.model.CropType.T, 
            self.model.CalendarType, 
            self.model.GrowingSeasonIndex.T,
            self.model.nFarm, self.model.nCrop, self.model.domain.nxy
            )        

        # ############################# #
        # TemperatureStress
        # ############################# #
        self.dynamic_temperature_stress()

        # ############################# #
        # BiomassAccumulation
        # ############################# #        
        aquacrop_fc.biomass_accumulation_w.update_biomass_accum_w(
            self.model.model.etref.values.T,
            self.model.TrAct.T,
            self.model.TrPot_NS.T,
            self.model.B.T,
            self.model.B_NS.T,
            self.model.BioTempStress.T,
            self.model.GDD.T,
            self.model.GDD_up.T,
            self.model.GDD_lo.T,
            self.model.PolHeatStress.T,
            self.model.model.tmax.values.T,
            self.model.Tmax_up.T,
            self.model.Tmax_lo.T,
            self.model.fshape_b.T,
            self.model.PolColdStress.T,
            self.model.model.tmin.values.T,
            self.model.Tmin_up.T,
            self.model.Tmin_lo.T,
            self.model.HI.T,
            self.model.PctLagPhase.T,
            self.model.YldFormCD.T,
            self.model.WP.T,
            self.model.WPy.T,
            self.model.fCO2.T,
            self.model.HIstartCD.T,
            self.model.DelayedCDs.T,
            self.model.DAP.T,
            self.model.CropType.T,
            self.model.Determinant.T,
            self.model.GrowingSeasonIndex.T,
            self.model.nFarm,
            self.model.nCrop,
            self.model.domain.nxy
        )
        
        # ############################# #
        # RootZoneWater
        # ############################# #
        
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.root_zone_water_w.update_root_zone_water_w(
            self.model.thRZ_Act.T, 
            self.model.thRZ_Sat.T, 
            self.model.thRZ_Fc.T, 
            self.model.thRZ_Wp.T, 
            self.model.thRZ_Dry.T, 
            self.model.thRZ_Aer.T, 
            self.model.TAW.T, 
            self.model.Dr.T, 
            self.model.th.T, 
            self.model.th_sat.T, 
            self.model.th_fc.T, 
            self.model.th_wilt.T, 
            self.model.th_dry.T, 
            self.model.Aer.T, 
            self.model.Zroot.T, 
            self.model.Zmin.T, 
            self.model.dz, 
            self.model.dz_sum, 
            layer_ix, 
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        )
        
        # ############################# #
        # WaterStress
        # ############################# #

        # TODO - convert to Fortran
        self.dynamic_water_stress(beta=True)

        # ############################# #
        # TemperatureStress
        # ############################# #
        self.dynamic_temperature_stress()

        # ############################# #
        # HarvestIndexAdjusted
        # ############################# #
        aquacrop_fc.harvest_index_w.adjust_harvest_index_w(
            self.model.HIadj.T,
            self.model.PreAdj.T,
            self.model.Fpre.T, 
            self.model.Fpol.T, 
            self.model.Fpost.T, 
            self.model.fpost_dwn.T, 
            self.model.fpost_upp.T, 
            self.model.sCor1.T, 
            self.model.sCor2.T,
            self.model.YieldForm.T,
            self.model.HI.T, 
            self.model.HI0.T, 
            self.model.dHI0.T, 
            self.model.B.T, 
            self.model.B_NS.T, 
            self.model.dHI_pre.T, 
            self.model.CC.T, 
            self.model.CCmin.T, 
            self.model.Ksw_Exp.T, 
            self.model.Ksw_Sto.T, 
            self.model.Ksw_Pol.T, 
            self.model.Kst_PolC.T, 
            self.model.Kst_PolH.T, 
            self.model.CanopyDevEndCD.T, 
            self.model.HIstartCD.T, 
            self.model.HIendCD.T, 
            self.model.YldFormCD.T, 
            self.model.FloweringCD.T, 
            self.model.a_HI.T, 
            self.model.b_HI.T, 
            self.model.exc.T, 
            self.model.DAP.T, 
            self.model.DelayedCDs.T, 
            self.model.CropType.T, 
            self.model.GrowingSeasonIndex.T, 
            self.model.nFarm, self.model.nCrop, self.model.domain.nxy
        )

        # ############################# #
        # CropYield
        # ############################# #
        self.dynamic_yield()

        # ############################# #
        # RootZoneWater
        # ############################# #
        
        layer_ix = self.model.layerIndex + 1
        aquacrop_fc.root_zone_water_w.update_root_zone_water_w(
            self.model.thRZ_Act.T, 
            self.model.thRZ_Sat.T, 
            self.model.thRZ_Fc.T, 
            self.model.thRZ_Wp.T, 
            self.model.thRZ_Dry.T, 
            self.model.thRZ_Aer.T, 
            self.model.TAW.T, 
            self.model.Dr.T, 
            self.model.th.T, 
            self.model.th_sat.T, 
            self.model.th_fc.T, 
            self.model.th_wilt.T, 
            self.model.th_dry.T, 
            self.model.Aer.T, 
            self.model.Zroot.T, 
            self.model.Zmin.T, 
            self.model.dz, 
            self.model.dz_sum, 
            layer_ix, 
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        )
        

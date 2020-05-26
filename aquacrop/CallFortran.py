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

    def dynamic(self):

        # # ############################# #
        # # GrowingDegreeDay
        # # ############################# #
        # aquacrop_fc.gdd_w.update_gdd_w(
        #     self.model.GDD.T, 
        #     self.model.GDDcum.T, 
        #     self.model.GDDmethod, 
        #     self.model.model.tmax.values.T, 
        #     self.model.model.tmin.values.T,
        #     self.model.Tbase.T,
        #     self.model.Tupp.T,
        #     self.model.GrowingSeasonIndex.T, 
        #     self.model.nFarm, self.model.nCrop, self.model.domain.nxy
        #     )
        
        # # ############################# #
        # # GrowthStage
        # # ############################# #
        # aquacrop_fc.growth_stage_w.update_growth_stage_w(
        #     np.int32(self.model.GrowthStage).T,
        #     self.model.Canopy10Pct.T,
        #     self.model.MaxCanopy.T,
        #     self.model.Senescence.T,
        #     self.model.GDDcum.T,
        #     np.int32(self.model.DAP).T,
        #     self.model.DelayedCDs.T,
        #     self.model.DelayedGDDs.T,
        #     int(self.model.CalendarType),
        #     self.model.GrowingSeasonIndex.T,
        #     self.model.nFarm,
        #     self.model.nCrop,
        #     self.model.domain.nxy
        # )        

        # # ############################# #
        # # Initial condition
        # # ############################# #

        # ****TODO: NOT YET IMPLEMENTED****
        
        # # Condition to identify crops which are not being grown or crops which
        # # have only just finished being grown. The water content of crops
        # # meeting this condition is used to compute the area-weighted initial
        # # condition
        # if np.any(self.model.GrowingSeasonDayOne):
        #     cond1 = np.logical_not(self.model.GrowingSeasonIndex) | self.model.GrowingSeasonDayOne
        #     cond1 = np.broadcast_to(cond1[:,:,None,:], self.model.th.shape)
        #     th = np.copy(self.model.th)
        #     th[(np.logical_not(cond1))] = np.nan

        #     # TEMPORARY FIX
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore", category=RuntimeWarning)
        #         th_ave = np.nanmean(th, axis=0) # average along farm dimension

        #     th_ave = np.broadcast_to(
        #         th_ave,
        #         (self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.domain.nxy)
        #     )
        #     cond2 = np.broadcast_to(self.model.GrowingSeasonDayOne[:,:,None,:], self.model.th.shape)
        #     self.model.th[cond2] = th_ave[cond2]
                
        # # ############################# #
        # # CheckGroundwaterTable
        # # ############################# #
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.check_gw_table_w.update_check_gw_table_w(
        #     self.model.th.T,
        #     self.model.th_fc_adj.T,
        #     np.int32(self.model.WTinSoil).T,
        #     self.model.th_sat.T,
        #     self.model.th_fc.T,
        #     int(self.model.groundwater.WaterTable),
        #     int(self.model.groundwater.DynamicWaterTable),
        #     self.model.groundwater.zGW,
        #     self.model.dz,
        #     layer_ix,
        #     self.model.nFarm,
        #     self.model.nCrop,
        #     self.model.nComp,
        #     self.model.nLayer,
        #     self.model.domain.nxy
        # )

        # # ############################# #
        # # PreIrrigation
        # # ############################# #        
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.pre_irr_w.update_pre_irr_w(
        #     self.model.PreIrr.T,
        #     self.model.th.T,
        #     self.model.IrrMethod.T,
        #     self.model.DAP.T,
        #     self.model.Zroot.T,
        #     self.model.Zmin.T,
        #     self.model.NetIrrSMT.T,
        #     self.model.th_fc.T,
        #     self.model.th_wilt.T,
        #     self.model.dz,
        #     self.model.dz_sum,
        #     layer_ix,
        #     self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        #     )

        # # ############################# #        
        # # Drainage
        # # ############################# #        
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.drainage_w.update_drainage_w(
        #     self.model.th.T,
        #     self.model.DeepPerc.T,
        #     self.model.FluxOut.T,
        #     self.model.th_sat.T,
        #     self.model.th_fc.T,
        #     self.model.k_sat.T,
        #     self.model.tau.T,
        #     self.model.th_fc_adj.T,
        #     self.model.dz,
        #     self.model.dz_sum,
        #     layer_ix,
        #     self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        #     )

        # # ############################# #
        # # RainfallPartition
        # # ############################# #
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.rainfall_partition_w.update_rain_part_w(
        #     self.model.Runoff.T,
        #     self.model.Infl.T,
        #     self.model.model.prec.values.T,
        #     # self.model.weather.precipitation.T,
        #     self.model.th.T,
        #     np.int32(self.model.DaySubmerged).T,
        #     np.int32(self.model.Bunds).T,
        #     self.model.zBund.T,
        #     self.model.th_fc.T,
        #     self.model.th_wilt.T,
        #     np.int32(self.model.CN).T,
        #     np.int32(self.model.adjustCurveNumber),
        #     self.model.zCN.T,
        #     self.model.CNbot.T,
        #     self.model.CNtop.T,
        #     self.model.dz,
        #     self.model.dz_sum,
        #     layer_ix,
        #     self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        #     )

        # # ############################# #
        # # RootZoneWater
        # # ############################# #
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.root_zone_water_w.update_root_zone_water_w(
        #     self.model.thRZ_Act.T, 
        #     self.model.thRZ_Sat.T, 
        #     self.model.thRZ_Fc.T, 
        #     self.model.thRZ_Wp.T, 
        #     self.model.thRZ_Dry.T, 
        #     self.model.thRZ_Aer.T, 
        #     self.model.TAW.T, 
        #     self.model.Dr.T, 
        #     self.model.th.T, 
        #     self.model.th_sat.T, 
        #     self.model.th_fc.T, 
        #     self.model.th_wilt.T, 
        #     self.model.th_dry.T, 
        #     self.model.Aer.T, 
        #     self.model.Zroot.T, 
        #     self.model.Zmin.T, 
        #     self.model.dz, 
        #     self.model.dz_sum, 
        #     layer_ix, 
        #     self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        # )

        # # ############################# #
        # # Irrigation
        # # ############################# #

        # aquacrop_fc.irrigation_w.update_irrigation_w(
        #     self.model.IrrMethod.T,
        #     self.model.Irr.T,      
        #     self.model.IrrCum.T,   
        #     self.model.IrrNetCum.T,
        #     self.model.SMT1.T,
        #     self.model.SMT2.T,
        #     self.model.SMT3.T,
        #     self.model.SMT4.T,
        #     self.model.IrrScheduled.T,  # TODO
        #     self.model.AppEff.T,
        #     self.model.Zroot.T,
        #     self.model.Zmin.T,
        #     self.model.TAW.T,
        #     self.model.Dr.T,
        #     self.model.thRZ_Fc.T,
        #     self.model.thRZ_Act.T,
        #     self.model.model.prec.values.T,
        #     self.model.Runoff.T,
        #     self.model.model.etref.values.T,
        #     self.model.MaxIrr.T,
        #     self.model.IrrInterval.T,
        #     self.model.DAP.T,
        #     self.model.GrowthStage.T,
        #     self.model.GrowingSeasonDayOne.T,
        #     self.model.GrowingSeasonIndex.T,
        #     int(self.model.nFarm),
        #     int(self.model.nCrop),
        #     int(self.model.domain.nxy)
        # )            
        
        # # ############################# #
        # # Infiltration
        # # ############################# #
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.infiltration_w.update_infl_w(
        #     self.model.Infl.T,
        #     self.model.SurfaceStorage.T,
        #     self.model.FluxOut.T,
        #     self.model.DeepPerc.T,
        #     self.model.Runoff.T,
        #     self.model.th.T,
        #     self.model.Irr.T,
        #     self.model.AppEff.T,
        #     self.model.Bunds.T,
        #     self.model.zBund.T,
        #     self.model.th_sat.T,
        #     self.model.th_fc.T,
        #     self.model.th_fc_adj.T,
        #     self.model.k_sat.T,
        #     self.model.tau.T,
        #     self.model.dz,
        #     layer_ix,
        #     self.model.nFarm,
        #     self.model.nCrop,
        #     self.model.nComp,
        #     self.model.nLayer,
        #     self.model.domain.nxy
        #     )        

        # # ############################# #
        # # CapillaryRise
        # # ############################# #
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.capillary_rise_w.update_cap_rise_w(
        #     self.model.CrTot.T,
        #     self.model.th.T,
        #     self.model.th_wilt.T,
        #     self.model.th_fc.T,
        #     self.model.th_fc_adj.T,
        #     self.model.k_sat.T,
        #     self.model.aCR.T,
        #     self.model.bCR.T,
        #     self.model.fshape_cr.T,
        #     self.model.FluxOut.T,
        #     self.model.groundwater.WaterTable,
        #     self.model.groundwater.zGW,
        #     self.model.dz,
        #     self.model.dz_layer,
        #     layer_ix,
        #     self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        #     )

        # # ############################# #
        # # Germination
        # # ############################# #
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.germination_w.update_germ_w(
        #     self.model.Germination.T,
        #     self.model.DelayedCDs.T,
        #     self.model.DelayedGDDs.T,
        #     self.model.GDD.T,
        #     self.model.th.T,
        #     self.model.th_fc.T,
        #     self.model.th_wilt.T,
        #     self.model.zGerm.T,
        #     self.model.GermThr.T,
        #     self.model.dz,
        #     self.model.dz_sum,
        #     layer_ix,
        #     self.model.GrowingSeasonIndex.T,
        #     self.model.nFarm,
        #     self.model.nCrop,
        #     self.model.nComp,
        #     self.model.nLayer,
        #     self.model.domain.nxy
        # )

        # ############################# #
        # RootDevelopment
        # ############################# #

        # # TODO - need to add reset_initial_conditions() to Fortran module
        # if np.any(self.model.GrowingSeasonDayOne):
        #     self.model.rCor[self.model.GrowingSeasonDayOne] = 1
        #     self.model.Zroot[self.model.GrowingSeasonDayOne] = self.model.Zmin[self.model.GrowingSeasonDayOne]

        # aquacrop_fc.root_dev_w.update_root_dev_w(
        #     self.model.Zroot.T, 
        #     self.model.rCor.T, 
        #     self.model.Zmin.T, 
        #     self.model.Zmax.T, 
        #     self.model.PctZmin.T, 
        #     self.model.Emergence.T, 
        #     self.model.MaxRooting.T, 
        #     self.model.fshape_r.T, 
        #     self.model.fshape_ex.T, 
        #     self.model.SxBot.T,
        #     self.model.SxTop.T,
        #     self.model.DAP.T,
        #     self.model.GDD.T,
        #     self.model.GDDcum.T,
        #     self.model.DelayedCDs.T,
        #     self.model.DelayedGDDs.T,
        #     self.model.TrRatio.T,
        #     self.model.Germination.T, 
        #     self.model.zRes.T,
        #     self.model.groundwater.WaterTable, 
        #     self.model.groundwater.zGW, 
        #     self.model.CalendarType, 
        #     self.model.GrowingSeasonIndex.T,
        #     self.model.nFarm, self.model.nCrop, self.model.domain.nxy
        # )

        # # ############################# #
        # # RootZoneWater
        # # ############################# #
        
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.root_zone_water_w.update_root_zone_water_w(
        #     self.model.thRZ_Act.T, 
        #     self.model.thRZ_Sat.T, 
        #     self.model.thRZ_Fc.T, 
        #     self.model.thRZ_Wp.T, 
        #     self.model.thRZ_Dry.T, 
        #     self.model.thRZ_Aer.T, 
        #     self.model.TAW.T, 
        #     self.model.Dr.T, 
        #     self.model.th.T, 
        #     self.model.th_sat.T, 
        #     self.model.th_fc.T, 
        #     self.model.th_wilt.T, 
        #     self.model.th_dry.T, 
        #     self.model.Aer.T, 
        #     self.model.Zroot.T, 
        #     self.model.Zmin.T, 
        #     self.model.dz, 
        #     self.model.dz_sum, 
        #     layer_ix, 
        #     self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        # )

                
        # # ############################# #
        # # WaterStress
        # # ############################# #

        # # # TODO - convert to Fortran
        # # self.dynamic_water_stress(beta=True)
        # aquacrop_fc.water_stress_w.update_water_stress_w(
        #     self.model.Ksw_Exp.T,
        #     self.model.Ksw_Sto.T,
        #     self.model.Ksw_Sen.T,
        #     self.model.Ksw_Pol.T,
        #     self.model.Ksw_StoLin.T,
        #     self.model.Dr.T,
        #     self.model.TAW.T,
        #     self.model.model.etref.values.T,
        #     # self.model.weather.referencePotET.T,
        #     self.model.ETadj.T,
        #     self.model.tEarlySen.T,
        #     self.model.p_up1.T,
        #     self.model.p_up2.T,
        #     self.model.p_up3.T,
        #     self.model.p_up4.T,
        #     self.model.p_lo1.T,
        #     self.model.p_lo2.T,
        #     self.model.p_lo3.T,
        #     self.model.p_lo4.T,
        #     self.model.fshape_w1.T,
        #     self.model.fshape_w2.T,
        #     self.model.fshape_w3.T,
        #     self.model.fshape_w4.T,
        #     int(True),
        #     self.model.nFarm,
        #     self.model.nCrop,
        #     self.model.domain.nxy
        # )        

        # # ############################# #        
        # # CanopyCover
        # # ############################# #
        # aquacrop_fc.canopy_cover_w.update_canopy_cover_w(
        #     self.model.CC.T,
        #     self.model.CCprev.T,
        #     self.model.CCadj.T,
        #     self.model.CC_NS.T,
        #     self.model.CCadj_NS.T,
        #     self.model.CCxW.T,
        #     self.model.CCxAct.T,
        #     self.model.CCxW_NS.T,
        #     self.model.CCxAct_NS.T,
        #     self.model.CC0adj.T,
        #     self.model.CCxEarlySen.T,
        #     self.model.tEarlySen.T,
        #     np.int32(self.model.PrematSenes).T,  # not required when all modules use Fortran
        #     self.model.CropDead.T,
        #     self.model.GDD.T,
        #     self.model.GDDcum.T,
        #     self.model.CC0.T,
        #     self.model.CCx.T,
        #     self.model.CGC.T,
        #     self.model.CDC.T,
        #     self.model.Emergence.T,
        #     self.model.Maturity.T,
        #     self.model.Senescence.T,
        #     self.model.CanopyDevEnd.T,
        #     self.model.Dr.T,
        #     self.model.TAW.T,
        #     self.model.model.etref.values.T,
        #     self.model.ETadj.T,
        #     self.model.p_up1.T,
        #     self.model.p_up2.T,
        #     self.model.p_up3.T,
        #     self.model.p_up4.T,
        #     self.model.p_lo1.T,
        #     self.model.p_lo2.T,
        #     self.model.p_lo3.T,
        #     self.model.p_lo4.T,
        #     self.model.fshape_w1.T,
        #     self.model.fshape_w2.T,
        #     self.model.fshape_w3.T,
        #     self.model.fshape_w4.T,
        #     self.model.GrowingSeasonIndex.T,
        #     self.model.GrowingSeasonDayOne.T,
        #     int(self.model.CalendarType),
        #     self.model.DAP.T,
        #     self.model.DelayedCDs.T,
        #     self.model.DelayedGDDs.T,
        #     int(self.model.nFarm),
        #     int(self.model.nCrop),
        #     int(self.model.domain.nxy)
        # )

        # # ############################# #
        # # SoilEvaporation
        # # ############################# #
        # self.model.EsAct = np.zeros(
        #     (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        # EvapTimeSteps = 20
        # prec = np.broadcast_to(self.model.model.prec.values,
        #                        (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        # etref = np.broadcast_to(self.model.model.etref.values,
        #                         (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        # EvapTimeSteps = 20
        # aquacrop_fc.soil_evaporation_w.update_soil_evap_w(
        #     np.float64(prec).T,
        #     np.float64(etref).T,
        #     self.model.EsAct.T,
        #     self.model.Epot.T,
        #     self.model.Irr.T,
        #     np.int32(self.model.IrrMethod).T,
        #     self.model.Infl.T,
        #     self.model.th.T,
        #     self.model.th_sat_comp.T,
        #     self.model.th_fc_comp.T,
        #     self.model.th_wilt_comp.T,
        #     self.model.th_dry_comp.T,
        #     self.model.SurfaceStorage.T,
        #     self.model.WetSurf.T,
        #     self.model.Wsurf.T,
        #     self.model.Wstage2.T,
        #     self.model.CC.T,
        #     self.model.CCadj.T,
        #     self.model.CCxAct.T,
        #     self.model.EvapZ.T,
        #     self.model.EvapZmin.T,
        #     self.model.EvapZmax.T,
        #     self.model.REW.T,
        #     self.model.Kex.T,
        #     self.model.CCxW.T,
        #     self.model.fwcc.T,
        #     self.model.fevap.T,
        #     self.model.fWrelExp.T,
        #     self.model.dz.T,
        #     self.model.dz_sum.T,
        #     np.int32(self.model.Mulches).T,
        #     self.model.fMulch.T,
        #     self.model.MulchPctGS.T,
        #     self.model.MulchPctOS.T,
        #     np.int32(self.model.GrowingSeasonIndex).T,
        #     np.int32(self.model.Senescence).T,
        #     np.int32(self.model.PrematSenes).T,
        #     np.int32(self.model.CalendarType),
        #     np.int32(self.model.DAP).T,
        #     np.int32(self.model.GDDcum).T,
        #     np.int32(self.model.DelayedCDs).T,
        #     np.int32(self.model.DelayedGDDs).T,
        #     self.model.model.time.timestep,
        #     EvapTimeSteps,
        #     self.model.nFarm,
        #     self.model.nCrop,
        #     self.model.nComp,
        #     self.model.domain.nxy
        # )

        # # ############################# #
        # # RootZoneWater
        # # ############################# #
        
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.root_zone_water_w.update_root_zone_water_w(
        #     self.model.thRZ_Act.T, 
        #     self.model.thRZ_Sat.T, 
        #     self.model.thRZ_Fc.T, 
        #     self.model.thRZ_Wp.T, 
        #     self.model.thRZ_Dry.T, 
        #     self.model.thRZ_Aer.T, 
        #     self.model.TAW.T, 
        #     self.model.Dr.T, 
        #     self.model.th.T, 
        #     self.model.th_sat.T, 
        #     self.model.th_fc.T, 
        #     self.model.th_wilt.T, 
        #     self.model.th_dry.T, 
        #     self.model.Aer.T, 
        #     self.model.Zroot.T, 
        #     self.model.Zmin.T, 
        #     self.model.dz, 
        #     self.model.dz_sum, 
        #     layer_ix, 
        #     self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        # )

        
        # # ############################# #
        # # WaterStress
        # # ############################# #

        # # # TODO - convert to Fortran
        # # self.dynamic_water_stress(beta=True)

        # aquacrop_fc.water_stress_w.update_water_stress_w(
        #     self.model.Ksw_Exp.T,
        #     self.model.Ksw_Sto.T,
        #     self.model.Ksw_Sen.T,
        #     self.model.Ksw_Pol.T,
        #     self.model.Ksw_StoLin.T,
        #     self.model.Dr.T,
        #     self.model.TAW.T,
        #     self.model.model.etref.values.T,
        #     # self.model.weather.referencePotET.T,
        #     self.model.ETadj.T,
        #     self.model.tEarlySen.T,
        #     self.model.p_up1.T,
        #     self.model.p_up2.T,
        #     self.model.p_up3.T,
        #     self.model.p_up4.T,
        #     self.model.p_lo1.T,
        #     self.model.p_lo2.T,
        #     self.model.p_lo3.T,
        #     self.model.p_lo4.T,
        #     self.model.fshape_w1.T,
        #     self.model.fshape_w2.T,
        #     self.model.fshape_w3.T,
        #     self.model.fshape_w4.T,
        #     int(True),
        #     self.model.nFarm,
        #     self.model.nCrop,
        #     self.model.domain.nxy
        # )
        
        # # ############################# #
        # # Transpiration
        # # ############################# #
        
        # # # reset initial conditions
        # # if np.any(self.model.GrowingSeasonDayOne):
        # #     self.model.AgeDays[self.model.GrowingSeasonDayOne] = 0  # not sure if required
        # #     self.model.AgeDays_NS[self.model.GrowingSeasonDayOne] = 0  # not sure if required
        # #     cond = self.model.GrowingSeasonDayOne
        # #     cond_comp = np.broadcast_to(cond[:,:,None,:], self.model.AerDaysComp.shape)
        # #     self.model.AerDays[cond] = 0
        # #     self.model.AerDaysComp[cond_comp] = 0        
        # #     self.model.Tpot[cond] = 0
        # #     self.model.TrRatio[cond] = 1
        # #     # self.model.TrAct[cond] = 0  # TEMP - may not require?
        # #     self.model.DaySubmerged[cond] = 0
        # #     # self.reset_initial_conditions()

        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.transpiration_w.update_transpiration_w(
        #     self.model.TrPot0.T, 
        #     self.model.TrPot_NS.T, 
        #     self.model.TrAct.T,
        #     self.model.TrAct0.T, 
        #     self.model.Tpot.T, 
        #     self.model.TrRatio.T,
        #     np.int32(self.model.AerDays).T, 
        #     np.int32(self.model.AerDaysComp).T, 
        #     self.model.th.T, 
        #     self.model.thRZ_Act.T, 
        #     self.model.thRZ_Sat.T, 
        #     self.model.thRZ_Fc.T,
        #     self.model.thRZ_Wp.T, 
        #     self.model.thRZ_Dry.T, 
        #     self.model.thRZ_Aer.T, 
        #     self.model.TAW.T, 
        #     self.model.Dr.T,
        #     np.int32(self.model.AgeDays).T,
        #     np.int32(self.model.AgeDays_NS).T,
        #     np.int32(self.model.DaySubmerged).T,
        #     self.model.SurfaceStorage.T, 
        #     self.model.IrrNet.T, 
        #     self.model.IrrNetCum.T, 
        #     self.model.CC.T, 
        #     self.model.model.etref.values.T, 
        #     # self.model.weather.referencePotET.T, 
        #     self.model.th_sat.T, 
        #     self.model.th_fc.T, 
        #     self.model.th_wilt.T, 
        #     self.model.th_dry.T, 
        #     np.int32(self.model.MaxCanopyCD).T, 
        #     self.model.Kcb.T, 
        #     self.model.Ksw_StoLin.T, 
        #     self.model.CCadj.T, 
        #     self.model.CCadj_NS.T,
        #     self.model.CCprev.T, 
        #     self.model.CCxW.T,
        #     self.model.CCxW_NS.T,
        #     self.model.Zroot.T,
        #     self.model.rCor.T,
        #     self.model.Zmin.T,
        #     self.model.a_Tr.T,
        #     self.model.Aer.T,
        #     self.model.fage.T, 
        #     np.int32(self.model.LagAer).T, 
        #     self.model.SxBot.T, 
        #     self.model.SxTop.T, 
        #     np.int32(self.model.ETadj).T,
        #     self.model.p_lo2.T, 
        #     self.model.p_up2.T, 
        #     self.model.fshape_w2.T, 
        #     np.int32(self.model.IrrMethod).T,
        #     self.model.NetIrrSMT.T,
        #     self.model.CurrentConc.T, 
        #     refconc,
        #     # refconc,
        #     np.int32(self.model.DAP).T,
        #     np.int32(self.model.DelayedCDs).T,
        #     self.model.dz.T, 
        #     self.model.dz_sum.T, 
        #     np.int32(layer_ix).T,
        #     np.int32(self.model.GrowingSeasonDayOne).T,
        #     np.int32(self.model.GrowingSeasonIndex).T,
        #     self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        #     )
        
        # ############################# #
        # AquaCrop Fortran module
        # ############################# #
        layer_ix = self.model.layerIndex + 1
        self.model.EsAct = np.zeros(
            (self.model.nFarm, self.model.nCrop, self.model.domain.nxy))
        self.model.GwIn = np.zeros((self.model.nCrop, self.model.nFarm, self.model.domain.nxy))
        EvapTimeSteps = 20
        mature = np.int32(self.model.CropMature).copy()        
        aquacrop_fc.aquacrop_w.update_aquacrop_w(
            self.model.GDD.T, 
            self.model.GDDcum.T, 
            self.model.GDDmethod, 
            self.model.model.tmax.values.T, 
            self.model.model.tmin.values.T,
            self.model.Tbase.T,
            self.model.Tupp.T,
            np.int32(self.model.GrowthStage).T,
            self.model.Canopy10Pct.T,
            self.model.MaxCanopy.T,
            self.model.Senescence.T,
            np.int32(self.model.DAP).T,
            self.model.DelayedCDs.T,
            self.model.DelayedGDDs.T,
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
            self.model.PreIrr.T,
            self.model.IrrMethod.T,
            self.model.Zroot.T,
            self.model.Zmin.T,
            self.model.NetIrrSMT.T,
            self.model.th_wilt.T,
            self.model.dz_sum,
            self.model.DeepPerc.T,
            self.model.FluxOut.T,
            self.model.k_sat.T,
            self.model.tau.T,
            self.model.Runoff.T,
            self.model.Infl.T,
            self.model.model.prec.values.T,
            np.int32(self.model.DaySubmerged).T,
            np.int32(self.model.Bunds).T,
            self.model.zBund.T,
            np.int32(self.model.CN).T,
            np.int32(self.model.adjustCurveNumber),
            self.model.zCN.T,
            self.model.CNbot.T,
            self.model.CNtop.T,
            self.model.thRZ_Act.T, 
            self.model.thRZ_Sat.T, 
            self.model.thRZ_Fc.T, 
            self.model.thRZ_Wp.T, 
            self.model.thRZ_Dry.T, 
            self.model.thRZ_Aer.T, 
            self.model.TAW.T, 
            self.model.Dr.T, 
            self.model.th_dry.T, 
            self.model.Aer.T,
            self.model.Irr.T,      
            self.model.IrrCum.T,   
            self.model.IrrNetCum.T,
            self.model.SMT1.T,
            self.model.SMT2.T,
            self.model.SMT3.T,
            self.model.SMT4.T,
            self.model.IrrScheduled.T,  # TODO
            self.model.AppEff.T,
            self.model.model.etref.values.T,
            self.model.MaxIrr.T,
            self.model.IrrInterval.T,            
            self.model.SurfaceStorage.T,
            self.model.CrTot.T,
            self.model.aCR.T,
            self.model.bCR.T,
            self.model.fshape_cr.T,
            self.model.dz_layer,
            self.model.Germination.T,
            self.model.zGerm.T,
            self.model.GermThr.T,
            self.model.rCor.T, 
            self.model.Zmax.T, 
            self.model.PctZmin.T, 
            self.model.Emergence.T, 
            self.model.MaxRooting.T, 
            self.model.fshape_r.T, 
            self.model.fshape_ex.T, 
            self.model.SxBot.T,
            self.model.SxTop.T,
            self.model.TrRatio.T,
            self.model.zRes.T,
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
            self.model.CC0.T,
            self.model.CCx.T,
            self.model.CGC.T,
            self.model.CDC.T,
            self.model.Maturity.T,
            self.model.CanopyDevEnd.T,
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
            self.model.EsAct.T,
            self.model.Epot.T,
            self.model.WetSurf.T,
            self.model.Wsurf.T,
            self.model.Wstage2.T,
            self.model.EvapZ.T,
            self.model.EvapZmin.T,
            self.model.EvapZmax.T,
            self.model.REW.T,
            self.model.Kex.T,
            self.model.CCxW.T,
            self.model.fwcc.T,
            self.model.fevap.T,
            self.model.fWrelExp.T,
            np.int32(self.model.Mulches).T,
            self.model.fMulch.T,
            self.model.MulchPctGS.T,
            self.model.MulchPctOS.T,
            self.model.model.time.timestep,
            EvapTimeSteps,
            self.model.TrPot0.T, 
            self.model.TrPot_NS.T, 
            self.model.TrAct.T,
            self.model.TrAct0.T, 
            self.model.Tpot.T, 
            np.int32(self.model.AerDays).T, 
            np.int32(self.model.AerDaysComp).T, 
            np.int32(self.model.AgeDays).T,
            np.int32(self.model.AgeDays_NS).T,
            np.int32(self.model.DaySubmerged).T,
            self.model.IrrNet.T, 
            np.int32(self.model.MaxCanopyCD).T, 
            self.model.Kcb.T, 
            self.model.CCxW_NS.T,
            self.model.a_Tr.T,
            self.model.fage.T, 
            np.int32(self.model.LagAer).T, 
            self.model.CurrentConc.T, 
            refconc,
            self.model.ETpot.T,
            self.model.GwIn.T,
            self.model.HI.T, 
            self.model.PctLagPhase.T,
            self.model.YieldForm.T,
            self.model.CCmin.T, 
            self.model.HIini.T, 
            self.model.HI0.T, 
            self.model.HIGC.T, 
            self.model.HIstart.T, 
            self.model.HIstartCD.T, 
            self.model.tLinSwitch.T, 
            self.model.dHILinear.T, 
            self.model.CropType.T,
            self.model.BioTempStress.T,
            self.model.GDD_up.T,
            self.model.GDD_lo.T,
            self.model.PolHeatStress.T,
            self.model.Tmax_up.T,
            self.model.Tmax_lo.T,
            self.model.fshape_b.T,
            self.model.PolColdStress.T,
            self.model.Tmin_up.T,
            self.model.Tmin_lo.T,
            self.model.B.T,
            self.model.B_NS.T,
            self.model.YldFormCD.T,
            self.model.WP.T,
            self.model.WPy.T,
            self.model.fCO2.T,
            self.model.Determinant.T,
            np.float64(self.model.HIadj).T,
            self.model.PreAdj.T,
            np.float64(self.model.Fpre).T, 
            np.float64(self.model.Fpol).T, 
            np.float64(self.model.Fpost).T, 
            np.float64(self.model.fpost_dwn).T, 
            np.float64(self.model.fpost_upp).T, 
            np.float64(self.model.sCor1).T, 
            np.float64(self.model.sCor2).T,
            self.model.dHI0.T, 
            self.model.dHI_pre.T, 
            self.model.CanopyDevEndCD.T, 
            self.model.HIendCD.T, 
            self.model.FloweringCD.T, 
            self.model.a_HI.T, 
            self.model.b_HI.T, 
            self.model.exc.T,
            self.model.Y.T,
            mature.T,
            int(self.model.CalendarType), self.model.GrowingSeasonDayOne.T, self.model.GrowingSeasonIndex.T,
            self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy            
        )                
        self.model.CropMature = mature.astype(bool).copy()

        # # ############################# #
        # # Evapotranspiration
        # # ############################# #

        # # TODO: add this to the end of transpiration module
        # self.model.ETpot = self.model.Epot + self.model.Tpot

        # # ############################# #
        # # Inflow
        # # ############################# #
        # self.model.GwIn = np.zeros((self.model.nCrop, self.model.nFarm, self.model.domain.nxy))
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.inflow_w.update_inflow_w(
        #     self.model.GwIn.T,
        #     self.model.th.T,
        #     np.int32(self.model.groundwater.WaterTable),
        #     self.model.groundwater.zGW.T,
        #     self.model.th_sat.T,
        #     self.model.dz,
        #     layer_ix,
        #     self.model.nFarm,
        #     self.model.nCrop,
        #     self.model.nComp,
        #     self.model.nLayer,
        #     self.model.domain.nxy
        #     )

        # # ############################# #
        # # HarvestIndex
        # # ############################# #
        # aquacrop_fc.harvest_index_w.update_harvest_index_w(
        #     self.model.HI.T, 
        #     self.model.PctLagPhase.T,
        #     self.model.YieldForm.T,
        #     self.model.CCprev.T, 
        #     self.model.CCmin.T, 
        #     self.model.CCx.T, 
        #     self.model.HIini.T, 
        #     self.model.HI0.T, 
        #     self.model.HIGC.T, 
        #     self.model.HIstart.T, 
        #     self.model.HIstartCD.T, 
        #     self.model.tLinSwitch.T, 
        #     self.model.dHILinear.T, 
        #     self.model.GDDcum.T, 
        #     self.model.DAP.T, 
        #     self.model.DelayedCDs.T, 
        #     self.model.DelayedGDDs.T, 
        #     self.model.CropType.T, 
        #     self.model.CalendarType, 
        #     self.model.GrowingSeasonIndex.T,
        #     self.model.nFarm, self.model.nCrop, self.model.domain.nxy
        #     )        

        # ############################# #
        # TemperatureStress
        # ############################# #
        # self.dynamic_temperature_stress()

        # # ############################# #
        # # BiomassAccumulation
        # # ############################# #        
        # aquacrop_fc.biomass_accumulation_w.update_biomass_accum_w(
        #     self.model.model.etref.values.T,
        #     self.model.TrAct.T,
        #     self.model.TrPot_NS.T,
        #     self.model.B.T,
        #     self.model.B_NS.T,
        #     self.model.BioTempStress.T,
        #     self.model.GDD.T,
        #     self.model.GDD_up.T,
        #     self.model.GDD_lo.T,
        #     self.model.PolHeatStress.T,
        #     self.model.model.tmax.values.T,
        #     self.model.Tmax_up.T,
        #     self.model.Tmax_lo.T,
        #     self.model.fshape_b.T,
        #     self.model.PolColdStress.T,
        #     self.model.model.tmin.values.T,
        #     self.model.Tmin_up.T,
        #     self.model.Tmin_lo.T,
        #     self.model.HI.T,
        #     self.model.PctLagPhase.T,
        #     self.model.YldFormCD.T,
        #     self.model.WP.T,
        #     self.model.WPy.T,
        #     self.model.fCO2.T,
        #     self.model.HIstartCD.T,
        #     self.model.DelayedCDs.T,
        #     self.model.DAP.T,
        #     self.model.CropType.T,
        #     self.model.Determinant.T,
        #     self.model.GrowingSeasonIndex.T,
        #     self.model.nFarm,
        #     self.model.nCrop,
        #     self.model.domain.nxy
        # )
        
        # # ############################# #
        # # RootZoneWater
        # # ############################# #
        
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.root_zone_water_w.update_root_zone_water_w(
        #     self.model.thRZ_Act.T, 
        #     self.model.thRZ_Sat.T, 
        #     self.model.thRZ_Fc.T, 
        #     self.model.thRZ_Wp.T, 
        #     self.model.thRZ_Dry.T, 
        #     self.model.thRZ_Aer.T, 
        #     self.model.TAW.T, 
        #     self.model.Dr.T, 
        #     self.model.th.T, 
        #     self.model.th_sat.T, 
        #     self.model.th_fc.T, 
        #     self.model.th_wilt.T, 
        #     self.model.th_dry.T, 
        #     self.model.Aer.T, 
        #     self.model.Zroot.T, 
        #     self.model.Zmin.T, 
        #     self.model.dz, 
        #     self.model.dz_sum, 
        #     layer_ix, 
        #     self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        # )
        
        # # ############################# #
        # # WaterStress
        # # ############################# #

        # # # TODO - convert to Fortran
        # # self.dynamic_water_stress(beta=True)

        # aquacrop_fc.water_stress_w.update_water_stress_w(
        #     self.model.Ksw_Exp.T,
        #     self.model.Ksw_Sto.T,
        #     self.model.Ksw_Sen.T,
        #     self.model.Ksw_Pol.T,
        #     self.model.Ksw_StoLin.T,
        #     self.model.Dr.T,
        #     self.model.TAW.T,
        #     self.model.model.etref.values.T,
        #     # self.model.weather.referencePotET.T,
        #     self.model.ETadj.T,
        #     self.model.tEarlySen.T,
        #     self.model.p_up1.T,
        #     self.model.p_up2.T,
        #     self.model.p_up3.T,
        #     self.model.p_up4.T,
        #     self.model.p_lo1.T,
        #     self.model.p_lo2.T,
        #     self.model.p_lo3.T,
        #     self.model.p_lo4.T,
        #     self.model.fshape_w1.T,
        #     self.model.fshape_w2.T,
        #     self.model.fshape_w3.T,
        #     self.model.fshape_w4.T,
        #     int(True),
        #     self.model.nFarm,
        #     self.model.nCrop,
        #     self.model.domain.nxy
        # )
        
        # # ############################# #
        # # TemperatureStress
        # # ############################# #
        # self.dynamic_temperature_stress()

        # # ############################# #
        # # HarvestIndexAdjusted
        # # ############################# #
        # aquacrop_fc.harvest_index_w.adjust_harvest_index_w(
        #     self.model.HIadj.T,
        #     self.model.PreAdj.T,
        #     self.model.Fpre.T, 
        #     self.model.Fpol.T, 
        #     self.model.Fpost.T, 
        #     self.model.fpost_dwn.T, 
        #     self.model.fpost_upp.T, 
        #     self.model.sCor1.T, 
        #     self.model.sCor2.T,
        #     self.model.YieldForm.T,
        #     self.model.HI.T, 
        #     self.model.HI0.T, 
        #     self.model.dHI0.T, 
        #     self.model.B.T, 
        #     self.model.B_NS.T, 
        #     self.model.dHI_pre.T, 
        #     self.model.CC.T, 
        #     self.model.CCmin.T, 
        #     self.model.Ksw_Exp.T, 
        #     self.model.Ksw_Sto.T, 
        #     self.model.Ksw_Pol.T,
        #     self.model.BioTempStress.T,
        #     self.model.GDD.T,
        #     self.model.GDD_up.T,
        #     self.model.GDD_lo.T,
        #     self.model.PolHeatStress.T,
        #     self.model.model.tmax.values.T,
        #     self.model.Tmax_up.T,
        #     self.model.Tmax_lo.T,
        #     self.model.fshape_b.T,
        #     self.model.PolColdStress.T,
        #     self.model.model.tmin.values.T,
        #     self.model.Tmin_up.T,
        #     self.model.Tmin_lo.T,
        #     # self.model.Kst_PolC.T, 
        #     # self.model.Kst_PolH.T, 
        #     self.model.CanopyDevEndCD.T, 
        #     self.model.HIstartCD.T, 
        #     self.model.HIendCD.T, 
        #     self.model.YldFormCD.T, 
        #     self.model.FloweringCD.T, 
        #     self.model.a_HI.T, 
        #     self.model.b_HI.T, 
        #     self.model.exc.T, 
        #     self.model.DAP.T, 
        #     self.model.DelayedCDs.T, 
        #     self.model.CropType.T, 
        #     self.model.GrowingSeasonIndex.T, 
        #     self.model.nFarm, self.model.nCrop, self.model.domain.nxy
        # )

        # # ############################# #
        # # CropYield
        # # ############################# #
        # # self.dynamic_yield()
        # mature = np.int32(self.model.CropMature).copy()        
        # aquacrop_fc.crop_yield_w.update_crop_yield_w(
        #     self.model.Y.T,
        #     mature.T,
        #     self.model.Maturity.T,
        #     self.model.B.T,
        #     self.model.HIadj.T,
        #     self.model.GDDcum.T,
        #     np.int32(self.model.GrowingSeasonIndex).T,
        #     np.int32(self.model.GrowingSeasonDayOne).T,
        #     int(self.model.CalendarType),
        #     np.int32(self.model.DAP).T,
        #     self.model.DelayedCDs.T,
        #     self.model.DelayedGDDs.T,
        #     int(self.model.nCrop),
        #     int(self.model.nFarm),
        #     int(self.model.domain.nxy)
        # )
        # self.model.CropMature = mature.astype(bool).copy()

        # # ############################# #
        # # RootZoneWater
        # # ############################# #
        
        # layer_ix = self.model.layerIndex + 1
        # aquacrop_fc.root_zone_water_w.update_root_zone_water_w(
        #     self.model.thRZ_Act.T, 
        #     self.model.thRZ_Sat.T, 
        #     self.model.thRZ_Fc.T, 
        #     self.model.thRZ_Wp.T, 
        #     self.model.thRZ_Dry.T, 
        #     self.model.thRZ_Aer.T, 
        #     self.model.TAW.T, 
        #     self.model.Dr.T, 
        #     self.model.th.T, 
        #     self.model.th_sat.T, 
        #     self.model.th_fc.T, 
        #     self.model.th_wilt.T, 
        #     self.model.th_dry.T, 
        #     self.model.Aer.T, 
        #     self.model.Zroot.T, 
        #     self.model.Zmin.T, 
        #     self.model.dz, 
        #     self.model.dz_sum, 
        #     layer_ix, 
        #     self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy
        # )
        

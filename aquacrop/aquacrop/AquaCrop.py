#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

from hm.model import Model
from hm.reporting import Reporting
from .io.Weather import Weather
from .io.Groundwater import Groundwater
# from .LandSurface import LandSurface
from .io.LandCoverParameters import AquaCropParameters
from .io.CarbonDioxide import CarbonDioxide
from .io.InitialCondition import InitialCondition
from .io import variable_list_crop
from .io.CarbonDioxide import refconc

import aquacrop_fc

import logging
logger = logging.getLogger(__name__)

class AquaCrop_old(Model):    
    def __init__(self, config, time, domain, init=None):
        super(AquaCrop, self).__init__(
            config,
            time,
            domain,
            is_1d=True,
            init=init
        )
        self.weather_module = Weather(self)
        self.groundwater_module = Groundwater(self)
        self.crop_module = LandSurface(self)
        
    def initial(self):
        self.weather_module.initial()
        self.groundwater_module.initial()
        self.crop_module.initial()
        
    def dynamic(self):
        self.weather_module.dynamic()
        self.groundwater_module.dynamic()
        self.crop_module.dynamic()

class AquaCrop(Model):
    def __init__(self, config, time, domain, init=None):        
        super(AquaCrop, self).__init__(
            config,
            time,
            domain,
            is_1d=True,
            init=init
        )
        self.weather_module = Weather(self)
        self.groundwater_module = Groundwater(self)
        self.carbon_dioxide_module = CarbonDioxide(self)
        self.lc_parameters_module = AquaCropParameters(self)        
        self.initial_condition_module = InitialCondition(self)
        
    def initial(self):
        self.weather_module.initial()
        self.groundwater_module.initial()
        self.carbon_dioxide_module.initial()
        self.lc_parameters_module.initial()
        self.initial_condition_module.initial()
        
        # TODO: take these out and put in 'constants.py' or similar
        state_vars = [
            'GDDcum', 'FluxOut', 'IrrCum', 'IrrNetCum'
        ]        
        int_surf_vars = [
            'WTinSoil', 'GrowthStage', 'DelayedCDs', 'Germination',
            'PrematSenes', 'CropDead', 'YieldForm', 'PreAdj',
            'CropMature', 'AgeDays', 'AgeDays_NS', 'AerDays',
            'DaySubmerged', 'PreAdj'            
        ]
        flt_surf_vars = [
            'GDD', 'GDDcum', 'PreIrr', 'IrrNet', 'DeepPerc',
            'Runoff', 'Infl', 'thRZ_Act', 'thRZ_Sat', 'thRZ_Fc',
            'thRZ_Wp', 'thRZ_Dry', 'thRZ_Aer', 'TAW', 'Dr', 'Wr',
            'Irr', 'IrrCum', 'IrrNetCum', 'CrTot', 'DelayedGDDs',
            'rCor', 'Zroot', 'Ksw_Exp', 'Ksw_Sto', 'Ksw_Sen',
            'Ksw_Pol', 'Ksw_StoLin', 'tEarlySen', 'CC', 'CCadj',
            'CC_NS', 'CCadj_NS', 'CCxAct', 'CCxAct_NS', 'CCxW',
            'CCxW_NS', 'CCxEarlySen', 'CCprev', 'CC0adj', 'Epot',
            'EvapZ', 'Wstage2', 'Wsurf', 'Wevap_Act', 'Wevap_Sat',
            'Wevap_Fc', 'Wevap_Wp', 'Wevap_Dry', 'EsAct', 'Ksa_Aer',
            'TrPot0', 'TrPot_NS', 'TrAct', 'TrAct0', 'Tpot', 'TrRatio',
            'ETpot', 'GwIn', 'HI', 'HIt', 'PctLagPhase', 'B', 'B_NS',
            'Fpre', 'Fpost', 'fpost_dwn', 'fpost_upp', 'Kst_Bio',
            'Kst_PolH', 'Kst_PolC', 'Fpol', 'sCor1', 'sCor2',
            'HIadj', 'Y'
        ]
        int_subsurf_vars = ['AerDaysComp']
        flt_subsurf_vars = ['FluxOut']
        initialize_to_one_vars = [
            'rCor', 'TrRatio', 'Fpre', 'Fpost', 'fpost_dwn',
            'fpost_upp'
        ]        
        int_vars = int_surf_vars + int_subsurf_vars
        flt_vars = flt_surf_vars + flt_subsurf_vars
        all_vars = int_vars + flt_vars        
        for varname in all_vars:
            if varname in int_surf_vars + int_subsurf_vars:
                datatype = np.int32
            else:
                datatype = np.float64
            if varname in int_subsurf_vars + flt_subsurf_vars:
                dims = (self.nFarm, self.nCrop, self.nComp, self.domain.nxy)
            else:
                dims = (self.nFarm, self.nCrop, self.domain.nxy)
            if varname in initialize_to_one_vars:
                init_val = 1
            else:
                init_val = 0                
            vars(self)[varname] = np.require(
                np.full(dims, init_val, dtype=datatype),
                requirements=['A','O','W','F']
            )                
        
    def dynamic(self):
        self.weather_module.dynamic()
        self.groundwater_module.dynamic()
        self.carbon_dioxide_module.dynamic(method='pad')
        self.lc_parameters_module.dynamic()
        # TODO:
        # self.initial_condition_module.dynamic()
        layer_ix = self.layerIndex + 1
        EvapTimeSteps = 20
        aquacrop_fc.aquacrop_w.update_aquacrop_w(
            self.GDD, 
            self.GDDcum, 
            self.GDDmethod, 
            self.tmax.values, 
            self.tmin.values,
            self.Tbase,
            self.Tupp,
            self.GrowthStage,
            self.Canopy10Pct,
            self.MaxCanopy,
            self.Senescence,
            self.DAP,
            self.DelayedCDs,
            self.DelayedGDDs,
            self.th,
            self.th_fc_adj,
            self.WTinSoil,
            self.th_sat,
            self.th_fc,
            int(self.groundwater_module.WaterTable),
            int(self.groundwater_module.DynamicWaterTable),
            self.groundwater_module.zGW,
            self.dz,
            layer_ix,
            self.PreIrr,
            self.IrrMethod,
            self.Zroot,
            self.Zmin,
            self.NetIrrSMT,
            self.th_wilt,
            self.dz_sum,
            self.DeepPerc,
            self.FluxOut,
            self.k_sat,
            self.tau,
            self.Runoff,
            self.Infl,
            self.prec.values,
            self.DaySubmerged,
            self.Bunds,
            self.zBund,
            self.CN,
            int(self.adjustCurveNumber),
            self.zCN,
            self.CNbot,
            self.CNtop,
            self.thRZ_Act, 
            self.thRZ_Sat, 
            self.thRZ_Fc, 
            self.thRZ_Wp, 
            self.thRZ_Dry, 
            self.thRZ_Aer, 
            self.TAW, 
            self.Dr, 
            self.th_dry, 
            self.Aer,
            self.Irr,      
            self.IrrCum,   
            self.IrrNetCum,
            self.SMT1,
            self.SMT2,
            self.SMT3,
            self.SMT4,
            self.IrrScheduled,  # TODO
            self.AppEff,
            self.etref.values,
            self.MaxIrr,
            self.IrrInterval,            
            self.SurfaceStorage,
            self.CrTot,
            self.aCR,
            self.bCR,
            self.fshape_cr,
            self.dz_layer,
            self.Germination,
            self.zGerm,
            self.GermThr,
            self.rCor, 
            self.Zmax, 
            self.PctZmin, 
            self.Emergence, 
            self.MaxRooting, 
            self.fshape_r, 
            self.fshape_ex, 
            self.SxBot,
            self.SxTop,
            self.TrRatio,
            self.zRes,
            self.CC,
            self.CCprev,
            self.CCadj,
            self.CC_NS,
            self.CCadj_NS,
            self.CCxW,
            self.CCxAct,
            self.CCxW_NS,
            self.CCxAct_NS,
            self.CC0adj,
            self.CCxEarlySen,
            self.tEarlySen,
            self.PrematSenes,  # not required when all modules use Fortran
            self.CropDead,
            self.CC0,
            self.CCx,
            self.CGC,
            self.CDC,
            self.Maturity,
            self.CanopyDevEnd,
            self.ETadj,
            self.p_up1,
            self.p_up2,
            self.p_up3,
            self.p_up4,
            self.p_lo1,
            self.p_lo2,
            self.p_lo3,
            self.p_lo4,
            self.fshape_w1,
            self.fshape_w2,
            self.fshape_w3,
            self.fshape_w4,
            self.EsAct,
            self.Epot,
            self.WetSurf,
            self.Wsurf,
            self.Wstage2,
            self.EvapZ,
            self.EvapZmin,
            self.EvapZmax,
            self.REW,
            self.Kex,
            self.fwcc,
            self.fevap,
            self.fWrelExp,
            self.Mulches,
            self.fMulch,
            self.MulchPctGS,
            self.MulchPctOS,
            self.time.timestep,
            EvapTimeSteps,
            self.TrPot0, 
            self.TrPot_NS, 
            self.TrAct,
            self.TrAct0, 
            self.Tpot, 
            self.AerDays, 
            self.AerDaysComp, 
            self.AgeDays,
            self.AgeDays_NS,
            self.DaySubmerged,
            self.IrrNet, 
            self.MaxCanopyCD, 
            self.Kcb, 
            self.a_Tr,
            self.fage, 
            self.LagAer, 
            self.CurrentConc, 
            refconc,
            self.ETpot,
            self.GwIn,
            self.HI, 
            self.PctLagPhase,
            self.YieldForm,
            self.CCmin, 
            self.HIini, 
            self.HI0, 
            self.HIGC, 
            self.HIstart, 
            self.HIstartCD, 
            self.tLinSwitch, 
            self.dHILinear, 
            self.CropType,
            self.BioTempStress,
            self.GDD_up,
            self.GDD_lo,
            self.PolHeatStress,
            self.Tmax_up,
            self.Tmax_lo,
            self.fshape_b,
            self.PolColdStress,
            self.Tmin_up,
            self.Tmin_lo,
            self.B,
            self.B_NS,
            self.YldFormCD,
            self.WP,
            self.WPy,
            self.fCO2,
            self.Determinant,
            self.HIadj,
            self.PreAdj,
            self.Fpre, 
            self.Fpol, 
            self.Fpost, 
            self.fpost_dwn, 
            self.fpost_upp, 
            self.sCor1, 
            self.sCor2,
            self.dHI0, 
            self.dHI_pre, 
            self.CanopyDevEndCD, 
            self.HIendCD, 
            self.FloweringCD, 
            self.a_HI, 
            self.b_HI, 
            self.exc,
            self.Y,
            self.CropMature,
            int(self.CalendarType),
            self.GrowingSeasonDayOne,
            self.GrowingSeasonIndex,
            self.nFarm,
            self.nCrop,
            self.nComp,
            self.nLayer,
            self.domain.nxy            
        )                
        # self.reporting_module.dynamic()
        

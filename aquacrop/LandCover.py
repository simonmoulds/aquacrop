#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

from hm.reporting import Reporting

from .io.LandCoverParameters import AquaCropParameters
from .io.CarbonDioxide import CarbonDioxide
from .io.InitialCondition import InitialCondition
from .io import variable_list_crop

from .io.CarbonDioxide import refconc

import aquacrop_fc


class LandCover(object):
    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.time = model.time
        self.domain = model.domain
        self.is_1d = model.is_1d
        self.weather = model.weather_module
        self.groundwater = model.groundwater_module
    def initial(self):
        pass
    def dynamic(self):
        pass


class Cropland(LandCover):
    def __init__(self, model):
        super(Cropland, self).__init__(model)
        self.carbon_dioxide_module = CarbonDioxide(self)
        self.lc_parameters_module = AquaCropParameters(self)        
        self.initial_condition_module = InitialCondition(self)

    def initial(self):
        self.carbon_dioxide_module.initial()
        self.lc_parameters_module.initial()
        self.initial_condition_module.initial()
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
            
        self.reporting_module = Reporting(
            self, variable_list_crop, 'CROP_PARAMETERS'
        )
        self.reporting_module.initial()

    def dynamic(self):        
        self.carbon_dioxide_module.dynamic(method='pad')
        self.lc_parameters_module.dynamic()
        layer_ix = self.layerIndex + 1
        EvapTimeSteps = 20
        aquacrop_fc.aquacrop_w.update_aquacrop_w(
            self.GDD, 
            self.GDDcum, 
            self.GDDmethod, 
            self.model.tmax.values, 
            self.model.tmin.values,
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
            int(self.groundwater.WaterTable),
            int(self.groundwater.DynamicWaterTable),
            self.groundwater.zGW,
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
            self.model.prec.values,
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
            self.model.etref.values,
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
            self.model.time.timestep,
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
        self.reporting_module.dynamic()

        # FOR DEBUGGING:
        
        # print('GDD:',self.GDD.shape)
        # print('GDDcum:',self.GDDcum.shape)
        # print('tmax:',self.model.tmax.values.shape) 
        # print('tmin:',self.model.tmin.values.shape)
        # print('Tbase:',self.Tbase.shape)
        # print('Tupp:',self.Tupp.shape)
        # print('GrowthStage:',self.GrowthStage.shape)
        # print('Canopy10Pct:',self.Canopy10Pct.shape)
        # print('MaxCanopy:',self.MaxCanopy.shape)
        # print('Senescence:',self.Senescence.shape)
        # print('DAP:',self.DAP.shape)        
        # print('DelayedCDs:',self.DelayedCDs.shape)
        # print('DelayedGDDs:',self.DelayedGDDs.shape)
        # print('th:',self.th.shape)
        # print('th_fc_adj:',self.th_fc_adj.shape)
        # print('WTinSoil:',self.WTinSoil.shape)
        # print('th_sat:',self.th_sat.shape)
        # print('th_fc:',self.th_fc.shape)
        # print('zGW:',self.groundwater.zGW.shape)
        # print('dz:',self.dz.shape)
        # print('PreIrr:',self.PreIrr.shape)
        # print('IrrMethod:',self.IrrMethod.shape)
        # print('Zroot:',self.Zroot.shape)
        # print('Zmin:',self.Zmin.shape)
        # print('NetIrrSMT:',self.NetIrrSMT.shape)
        # print('th_wilt:',self.th_wilt.shape)
        # print('dz_sum:',self.dz_sum.shape)
        # print('DeepPerc',self.DeepPerc.shape)
        # print('FluxOut',self.FluxOut.shape)
        # print('k_sat',self.k_sat.shape)
        # print('tau',self.tau.shape)
        # print('runoff',self.Runoff.shape)
        # print('infl',self.Infl.shape)
        # print('prec',self.model.prec.values.shape)
        # print('daysubmerged',self.DaySubmerged.shape)
        # print('bunds',self.Bunds.shape)
        # print('zbund',self.zBund.shape)
        # print('cn',self.CN.shape)
        # print('zcn',self.zCN.shape)
        # print('cnbot',self.CNbot.shape)
        # print('cntop',self.CNtop.shape)
        # print('thrz_act',self.thRZ_Act.shape) 
        # print('thrz_sat',self.thRZ_Sat.shape) 
        # print('thrz_fc',self.thRZ_Fc.shape) 
        # print('thrz_wp',self.thRZ_Wp.shape) 
        # print('thrz_dry',self.thRZ_Dry.shape) 
        # print('thrz_aer',self.thRZ_Aer.shape) 
        # print('taw',self.TAW.shape) 
        # print('dr',self.Dr.shape) 
        # print('th_dry:',self.th_dry.shape) 
        # print('aer:',self.Aer.shape)
        # print('irr:',self.Irr.shape)      
        # print('irrcum:',self.IrrCum.shape)   
        # print('irrnetcum:',self.IrrNetCum.shape)
        # print('smt1:',self.SMT1.shape)
        # print('smt2:',self.SMT2.shape)
        # print('smt3:',self.SMT3.shape)
        # print('smt4:',self.SMT4.shape)
        # print('irrscheduled:',self.IrrScheduled.shape)  # TODO
        # print('appeff:',self.AppEff.shape)
        # print('etref:',self.model.etref.values.shape)
        # print('maxirr:',self.MaxIrr.shape)
        # print('irrinterval:',self.IrrInterval.shape)            
        # print('surfstor:',self.SurfaceStorage.shape)
        # print('crtot:',self.CrTot.shape)
        # print('acr:',self.aCR.shape)
        # print('bcr:',self.bCR.shape)
        # print('fshape_cr:',self.fshape_cr.shape)
        # print('dz_lyr:',self.dz_layer.shape)
        # print('germination:',self.Germination.shape)
        # print('zgerm:',self.zGerm.shape)
        # print('germthr:',self.GermThr.shape)
        # print('rcor:',self.rCor.shape) 
        # print('zmax:',self.Zmax.shape) 
        # print('pctzmin:',self.PctZmin.shape) 
        # print('emergence:',self.Emergence.shape) 
        # print('maxrooting:',self.MaxRooting.shape) 
        # print('fshape_r:',self.fshape_r.shape) 
        # print('fshape_ex:',self.fshape_ex.shape) 
        # print('sxbot:',self.SxBot.shape)
        # print('sxtop:',self.SxTop.shape)
        # print('trratio:',self.TrRatio.shape)
        # print('zres:',self.zRes.shape)
        # print('cc:',self.CC.shape)
        # print('ccprev:',self.CCprev.shape)
        # print('ccadj:',self.CCadj.shape)
        # print('ccns:',self.CC_NS.shape)
        # print('ccadjns:',self.CCadj_NS.shape)
        # print('ccxw:',self.CCxW.shape)
        # print('ccxact:',self.CCxAct.shape)
        # print('ccxwns:',self.CCxW_NS.shape)
        # print('ccxactns:',self.CCxAct_NS.shape)
        # print('cc0adj:',self.CC0adj.shape)
        # print('ccxearlysen:',self.CCxEarlySen.shape)
        # print('tearlysen:',self.tEarlySen.shape)
        # print('prematsenes:',self.PrematSenes.shape) # NOT REQD
        # print('cropdead:',self.CropDead.shape)
        # print('cc0:',self.CC0.shape)
        # print('ccx:',self.CCx.shape)
        # print('cgc:',self.CGC.shape)
        # print('cdc:',self.CDC.shape)
        # print('maturity:',self.Maturity.shape)
        # print('canopydevend:',self.CanopyDevEnd.shape)
        # print('etadj:',self.ETadj.shape)
        # print('pup1:',self.p_up1.shape)
        # print('pup2:',self.p_up2.shape)
        # print('pup3:',self.p_up3.shape)
        # print('pup4:',self.p_up4.shape)
        # print('plo1:',self.p_lo1.shape)
        # print('plo2:',self.p_lo2.shape)
        # print('plo3:',self.p_lo3.shape)
        # print('plo4:',self.p_lo4.shape)
        # print('fshapew1:',self.fshape_w1.shape)
        # print('fshapew2:',self.fshape_w2.shape)
        # print('fshapew3:',self.fshape_w3.shape)
        # print('fshapew4:',self.fshape_w4.shape)
        # print('esact:',self.EsAct.shape)
        # print('epot:',self.Epot.shape)
        # print('wetsurf:',self.WetSurf.shape)
        # print('wsurf:',self.Wsurf.shape)
        # print('wstage2:',self.Wstage2.shape)
        # print('evapz:',self.EvapZ.shape)
        # print('evapzmin:',self.EvapZmin.shape)
        # print('evapzmax:',self.EvapZmax.shape)
        # print('rew:',self.REW.shape)
        # print('kex:',self.Kex.shape)
        # print('fwcc:',self.fwcc.shape)
        # print('fevap:',self.fevap.shape)
        # print('fwrelexp:',self.fWrelExp.shape)
        # print('mulches:',self.Mulches.shape)
        # print('fmulch:',self.fMulch.shape)
        # print('mulchpctgs:',self.MulchPctGS.shape)
        # print('mulchpctos:',self.MulchPctOS.shape)
        # print('trpot0:',self.TrPot0.shape) 
        # print('trpotns:',self.TrPot_NS.shape) 
        # print('tract:',self.TrAct.shape)
        # print('tract0:',self.TrAct0.shape) 
        # print('tpot:',self.Tpot.shape) 
        # print('aerdays:',self.AerDays.shape) 
        # print('aerdayscomp:',self.AerDaysComp.shape) 
        # print('agedays:',self.AgeDays.shape)
        # print('agedays_ns:',self.AgeDays_NS.shape)
        # print('daysubmerged:',self.DaySubmerged.shape)
        # print('irrnet:',self.IrrNet.shape) 
        # print('maxcanopycd:',self.MaxCanopyCD.shape) 
        # print('kcb:',self.Kcb.shape) 
        # print('atr:',self.a_Tr.shape)
        # print('fage:',self.fage.shape) 
        # print('lagaer:',self.LagAer.shape) 
        # print('CurrentConc:',self.CurrentConc.shape) 
        # print('etpot:',self.ETpot.shape)
        # print('gwin:',self.GwIn.shape)
        # print('hi:',self.HI.shape) 
        # print('pctlagphase:',self.PctLagPhase.shape)
        # print('yieldform:',self.YieldForm.shape)
        # print('ccmin:',self.CCmin.shape) 
        # print('hiini:',self.HIini.shape) 
        # print('hi0:',self.HI0.shape) 
        # print('higc:',self.HIGC.shape) 
        # print('histart:',self.HIstart.shape) 
        # print('histartcd:',self.HIstartCD.shape) 
        # print('tlinswitch:',self.tLinSwitch.shape) 
        # print('dhilinear:',self.dHILinear.shape) 
        # print('croptype:',self.CropType.shape)
        # print('biotempstress:',self.BioTempStress.shape)
        # print('gddup:',self.GDD_up.shape)
        # print('gddlo:',self.GDD_lo.shape)
        # print('polheatstress:',self.PolHeatStress.shape)
        # print('tmax_up:',self.Tmax_up.shape)
        # print('tmax_lo:',self.Tmax_lo.shape)
        # print('fshape_b:',self.fshape_b.shape)
        # print('polcoldstress:',self.PolColdStress.shape)
        # print('tmin_up:',self.Tmin_up.shape)
        # print('tmin_lo:',self.Tmin_lo.shape)
        # print('b:',self.B.shape)
        # print('b_ns:',self.B_NS.shape)
        # print('yldformcd:',self.YldFormCD.shape)
        # print('wp:',self.WP.shape)
        # print('wpy:',self.WPy.shape)
        # print('fco2:',self.fCO2.shape)
        # print('determinant:',self.Determinant.shape)
        # print('hiadj:',self.HIadj.shape)
        # print('preadj:',self.PreAdj.shape)
        # print('fpre:',self.Fpre.shape) 
        # print('fpol:',self.Fpol.shape) 
        # print('fpost:',self.Fpost.shape) 
        # print('fpost_dwn:',self.fpost_dwn.shape) 
        # print('fpost_upp:',self.fpost_upp.shape) 
        # print('scor1:',self.sCor1.shape) 
        # print('scor2:',self.sCor2.shape)
        # print('dhi0:',self.dHI0.shape) 
        # print('dhi_pre:',self.dHI_pre.shape) 
        # print('canopydevend:',self.CanopyDevEndCD.shape) 
        # print('hiendcd:',self.HIendCD.shape) 
        # print('flowering:',self.FloweringCD.shape) 
        # print('ahi:',self.a_HI.shape) 
        # print('bhi:',self.b_HI.shape) 
        # print('exc:',self.exc.shape)
        # print('y:',self.Y.shape)
        # print('cropmature:',self.CropMature.shape)
        # print('gsday1:',self.GrowingSeasonDayOne.shape)
        # print('gsix:',self.GrowingSeasonIndex.shape)

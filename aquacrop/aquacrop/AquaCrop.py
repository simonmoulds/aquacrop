#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import netCDF4 as nc
import datetime as datetime
import calendar as calendar

from hm.dynamicmodel import HmEnKfModel
from hm.model import Model
from hm.reporting import Reporting
from .io.Weather import Weather
from .io.Groundwater import Groundwater
from .io.LandCoverParameters import AquaCropParameters
from .io.CarbonDioxide import CarbonDioxide
from .io.InitialCondition import InitialCondition
from .io import variable_list_crop
from .io.CarbonDioxide import refconc

import aquacrop_fc

import logging
logger = logging.getLogger(__name__)


class AqEnKfModel(HmEnKfModel):
    # To begin with we consider the simplest case, where the model is running for one point in space
    #
    # Input we need:
    # Observed canopy cover (as textfile)
    def setState(self):
        modelled_canopy_cover = np.array((self.model.CC[0, 0, 0],))
        return modelled_canopy_cover

    def setObservations(self):
        timestep = self.currentTimeStep()
        # The time points for which observed data is available is defined by setFilterTimesteps method (currently hard-coded in the cli script)
        fn = 'obs' + str(timestep) + '.txt'
        with open(fn) as f:
            obs_canopy_cover = [float(val) for val in f.read().split()]
        obs_canopy_cover = np.array(
            [obs_canopy_cover, ] * self.nrSamples()).transpose()

        # TODO: work out appropriate way to estimate covariance
        covariance = np.random.random((1, 1))
        # covariance = np.zeros((1, 1))
        # covariance = np.ones((1, 1)) * 0.005
        self.setObservedMatrices(obs_canopy_cover, covariance)

    def postmcloop(self):
        # used to calculate statistics of the ensemble (e.g. mean, variance, percentiles)
        # TODO:
        # * Define variables of interest
        # * Calculate average, variance - DONE
        # * Calculate percentiles
        self.reporting.create_mc_summary_variable()

    def resume(self):
        HmEnKfModel.resume(self)
        updated_canopy_cover = self.getStateVector(self.currentSampleNumber())
        self.model.CC[0, 0, 0] = updated_canopy_cover


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
        self.state_varnames = [
            'DAP',
            'GDDcum', 'th', 'DaySubmerged', 'IrrCum', 'IrrNetCum',
            'SurfaceStorage', 'Germination', 'DelayedCDs', 'DelayedGDDs',
            'Zroot', 'CC', 'CC_NS', 'CCxAct', 'CCxAct_NS', 'CCxW', 'CCxW_NS',
            'CCxEarlySen', 'CCprev', 'CC0adj', 'tEarlySen', 'PrematSenes',
            'CropDead', 'CCadj', 'CCadj_NS', 'Wsurf', 'EvapZ', 'AerDays',
            'AerDaysComp', 'AgeDays', 'AgeDays_NS', 'HI', 'PctLagPhase',
            'B', 'B_NS', 'HIadj', 'PreAdj', 'Fpre', 'Fpost', 'fpost_dwn',
            'fpost_upp', 'Fpol', 'sCor1', 'sCor2', 'CropMature'
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
                requirements=['A', 'O', 'W', 'F']
            )

        # # TEMPORARY FIX
        # self.P = self.prec.values

    def dynamic(self):
        self.weather_module.dynamic()
        self.groundwater_module.dynamic()
        self.carbon_dioxide_module.dynamic(method='pad')
        self.lc_parameters_module.dynamic()

        # # TEMPORARY FIX
        # self.P = self.prec.values

        # TODO:
        # self.initial_condition_module.dynamic()
        layer_ix = self.layerIndex + 1
        EvapTimeSteps = 20
        # print('B    : ', self.B)
        # print('HIadj: ', self.HIadj)
        # print('GrowingSeason: ', self.GrowingSeasonIndex)
        # print('YieldForm: ', self.YieldForm)
        # print('HIt:', self.HIt)

        aquacrop_fc.aquacrop_w.update_aquacrop_w(
            self.GDD,
            self.GDDcum,
            self.GDDmethod,
            self.Tmax,
            self.Tmin,
            # self.tmax.values,
            # self.tmin.values,
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
            # self.prec.values,
            self.P,
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
            # self.etref.values,
            self.ETref,
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

        # # print('PlantingDate:', self.PlantingDate[...,0])
        # # print('HarvestDate:', self.HarvestDate[...,0])
        # # print('GDD:',self.GDD[...,0])
        # # print('GDDcum:',self.GDDcum[...,0])
        # # print('tmax:',self.model.tmax.values)
        # # print('tmin:',self.model.tmin.values)
        # # print('Tbase:',self.Tbase[...,0])
        # # print('Tupp:',self.Tupp[...,0])
        # # print('GrowthStage:',self.GrowthStage[...,0])
        # # print('Canopy10Pct:',self.Canopy10Pct[...,0])
        # # print('MaxCanopy:',self.MaxCanopy[...,0])
        # # print('Senescence:',self.Senescence[...,0])
        # # print('DAP:',self.DAP[...,0])
        # # print('DelayedCDs:',self.DelayedCDs[...,0])
        # # print('DelayedGDDs:',self.DelayedGDDs[...,0])
        # # print('th:',self.th[...,0])
        # # print('th_fc_adj:',self.th_fc_adj[...,0])
        # # print('WTinSoil:',self.WTinSoil[...,0])
        # # print('th_sat:',self.th_sat[...,0])
        # # print('th_fc:',self.th_fc[...,0])
        # # print('zGW:',self.groundwater.zGW[...,0])
        # # print('dz:',self.dz[...,0])
        # # print('PreIrr:',self.PreIrr[...,0])
        # # print('IrrMethod:',self.IrrMethod[...,0])
        # # print('Zroot:',self.Zroot[...,0])
        # # print('Zmin:',self.Zmin[...,0])
        # # print('NetIrrSMT:',self.NetIrrSMT[...,0])
        # # print('th_wilt:',self.th_wilt[...,0])
        # # print('dz_sum:',self.dz_sum[...,0])
        # # print('DeepPerc',self.DeepPerc[...,0])
        # # print('FluxOut',self.FluxOut[...,0])
        # # print('k_sat',self.k_sat[...,0])
        # # print('tau',self.tau[...,0])
        # # print('runoff',self.Runoff[...,0])
        # # print('infl',self.Infl[...,0])
        # # print('prec',self.model.prec.values)
        # # print('daysubmerged',self.DaySubmerged[...,0])
        # # print('bunds',self.Bunds[...,0])
        # # print('zbund',self.zBund[...,0])
        # # print('cn',self.CN[...,0])
        # # print('zcn',self.zCN[...,0])
        # # print('cnbot',self.CNbot[...,0])
        # # print('cntop',self.CNtop[...,0])
        # # print('thrz_act',self.thRZ_Act[...,0])
        # # print('thrz_sat',self.thRZ_Sat[...,0])
        # # print('thrz_fc',self.thRZ_Fc[...,0])
        # # print('thrz_wp',self.thRZ_Wp[...,0])
        # # print('thrz_dry',self.thRZ_Dry[...,0])
        # # print('thrz_aer',self.thRZ_Aer[...,0])
        # # print('taw',self.TAW[...,0])
        # # print('dr',self.Dr[...,0])
        # # print('th_dry:',self.th_dry[...,0])
        # # print('aer:',self.Aer[...,0])
        # # print('irr:',self.Irr[...,0])
        # # print('irrcum:',self.IrrCum[...,0])
        # # print('irrnetcum:',self.IrrNetCum[...,0])
        # # print('smt1:',self.SMT1[...,0])
        # # print('smt2:',self.SMT2[...,0])
        # # print('smt3:',self.SMT3[...,0])
        # # print('smt4:',self.SMT4[...,0])
        # # print('irrscheduled:',self.IrrScheduled[...,0])  # TODO
        # # print('appeff:',self.AppEff[...,0])
        # # print('etref:',self.model.etref.values[...,0])
        # # print('maxirr:',self.MaxIrr[...,0])
        # # print('irrinterval:',self.IrrInterval[...,0])
        # # print('surfstor:',self.SurfaceStorage[...,0])
        # # print('crtot:',self.CrTot[...,0])
        # # print('acr:',self.aCR[...,0])
        # # print('bcr:',self.bCR[...,0])
        # # print('fshape_cr:',self.fshape_cr[...,0])
        # # print('dz_lyr:',self.dz_layer[...,0])
        # # print('germination:',self.Germination[...,0])
        # # print('zgerm:',self.zGerm[...,0])
        # # print('germthr:',self.GermThr[...,0])
        # # print('rcor:',self.rCor[...,0])
        # # print('zmax:',self.Zmax[...,0])
        # # print('pctzmin:',self.PctZmin[...,0])
        # # print('emergence:',self.Emergence[...,0])
        # # print('maxrooting:',self.MaxRooting[...,0])
        # # print('fshape_r:',self.fshape_r[...,0])
        # # print('fshape_ex:',self.fshape_ex[...,0])
        # # print('sxbot:',self.SxBot[...,0])
        # # print('sxtop:',self.SxTop[...,0])
        # # print('trratio:',self.TrRatio[...,0])
        # # print('zres:',self.zRes[...,0])
        # # print('cc:',self.CC[...,0])
        # # print('ccprev:',self.CCprev[...,0])
        # # print('ccadj:',self.CCadj[...,0])
        # # print('ccns:',self.CC_NS[...,0])
        # # print('ccadjns:',self.CCadj_NS[...,0])
        # # print('ccxw:',self.CCxW[...,0])
        # # print('ccxact:',self.CCxAct[...,0])
        # # print('ccxwns:',self.CCxW_NS[...,0])
        # # print('ccxactns:',self.CCxAct_NS[...,0])
        # # print('cc0adj:',self.CC0adj[...,0])
        # # print('ccxearlysen:',self.CCxEarlySen[...,0])
        # # print('tearlysen:',self.tEarlySen[...,0])
        # # print('prematsenes:',self.PrematSenes[...,0]) # NOT REQD
        # # print('cropdead:',self.CropDead[...,0])
        # # print('cc0:',self.CC0[...,0])
        # # print('ccx:',self.CCx[...,0])
        # # print('cgc:',self.CGC[...,0])
        # # print('cdc:',self.CDC[...,0])
        # # print('maturity:',self.Maturity[...,0])
        # # print('canopydevend:',self.CanopyDevEnd[...,0])
        # # print('etadj:',self.ETadj[...,0])
        # # print('pup1:',self.p_up1[...,0])
        # # print('pup2:',self.p_up2[...,0])
        # # print('pup3:',self.p_up3[...,0])
        # # print('pup4:',self.p_up4[...,0])
        # # print('plo1:',self.p_lo1[...,0])
        # # print('plo2:',self.p_lo2[...,0])
        # # print('plo3:',self.p_lo3[...,0])
        # # print('plo4:',self.p_lo4[...,0])
        # # print('fshapew1:',self.fshape_w1[...,0])
        # # print('fshapew2:',self.fshape_w2[...,0])
        # # print('fshapew3:',self.fshape_w3[...,0])
        # # print('fshapew4:',self.fshape_w4[...,0])
        # # print('esact:',self.EsAct[...,0])
        # # print('epot:',self.Epot[...,0])
        # # print('wetsurf:',self.WetSurf[...,0])
        # # print('wsurf:',self.Wsurf[...,0])
        # # print('wstage2:',self.Wstage2[...,0])
        # # print('evapz:',self.EvapZ[...,0])
        # # print('evapzmin:',self.EvapZmin[...,0])
        # # print('evapzmax:',self.EvapZmax[...,0])
        # # print('rew:',self.REW[...,0])
        # # print('kex:',self.Kex[...,0])
        # # print('fwcc:',self.fwcc[...,0])
        # # print('fevap:',self.fevap[...,0])
        # # print('fwrelexp:',self.fWrelExp[...,0])
        # # print('mulches:',self.Mulches[...,0])
        # # print('fmulch:',self.fMulch[...,0])
        # # print('mulchpctgs:',self.MulchPctGS[...,0])
        # # print('mulchpctos:',self.MulchPctOS[...,0])
        # # print('trpot0:',self.TrPot0[...,0])
        # # print('trpotns:',self.TrPot_NS[...,0])
        # # print('tract:',self.TrAct[...,0])
        # # print('tract0:',self.TrAct0[...,0])
        # # print('tpot:',self.Tpot[...,0])
        # # print('prec:',self.model.prec.values[...,0])
        # # print('runoff:',self.Runoff[...,0])
        # # print('th:',self.th[...,0])
        # # print('aerdays:',self.AerDays[...,0])
        # # print('aerdayscomp:',self.AerDaysComp[...,0])
        # # print('agedays:',self.AgeDays[...,0])
        # # print('agedays_ns:',self.AgeDays_NS[...,0])
        # # print('daysubmerged:',self.DaySubmerged[...,0])
        # # print('irrnet:',self.IrrNet[...,0])
        # # print('maxcanopycd:',self.MaxCanopyCD[...,0])
        # # print('kcb:',self.Kcb[...,0])
        # # print('atr:',self.a_Tr[...,0])
        # # print('fage:',self.fage[...,0])
        # # print('lagaer:',self.LagAer[...,0])
        # # print('CurrentConc:',self.CurrentConc[...,0])
        # # print('etpot:',self.ETpot[...,0])
        # # print('gwin:',self.GwIn[...,0])
        # # print('hi:',self.HI[...,0])
        # # print('pctlagphase:',self.PctLagPhase[...,0])
        # # print('yieldform:',self.YieldForm[...,0])
        # # print('ccmin:',self.CCmin[...,0])
        # # print('hiini:',self.HIini[...,0])
        # # print('hi0:',self.HI0[...,0])
        # # print('higc:',self.HIGC[...,0])
        # # print('histart:',self.HIstart[...,0])
        # # print('histartcd:',self.HIstartCD[...,0])
        # # print('tlinswitch:',self.tLinSwitch[...,0])
        # # print('dhilinear:',self.dHILinear[...,0])
        # # print('croptype:',self.CropType[...,0])
        # # print('biotempstress:',self.BioTempStress[...,0])
        # # print('gddup:',self.GDD_up[...,0])
        # # print('gddlo:',self.GDD_lo[...,0])
        # # print('polheatstress:',self.PolHeatStress[...,0])
        # # print('tmax_up:',self.Tmax_up[...,0])
        # # print('tmax_lo:',self.Tmax_lo[...,0])
        # # print('fshape_b:',self.fshape_b[...,0])
        # # print('polcoldstress:',self.PolColdStress[...,0])
        # # print('tmin_up:',self.Tmin_up[...,0])
        # # print('tmin_lo:',self.Tmin_lo[...,0])
        # # print('b:',self.B[...,0])
        # # print('b_ns:',self.B_NS[...,0])
        # # print('yldformcd:',self.YldFormCD[...,0])
        # # print('wp:',self.WP[...,0])
        # # print('wpy:',self.WPy[...,0])
        # # print('fco2:',self.fCO2[...,0])
        # # print('determinant:',self.Determinant[...,0])
        # # print('hiadj:',self.HIadj[...,0])
        # # print('preadj:',self.PreAdj[...,0])
        # # print('fpre:',self.Fpre[...,0])
        # # print('fpol:',self.Fpol[...,0])
        # # print('fpost:',self.Fpost[...,0])
        # # print('fpost_dwn:',self.fpost_dwn[...,0])
        # # print('fpost_upp:',self.fpost_upp[...,0])
        # # print('scor1:',self.sCor1[...,0])
        # # print('scor2:',self.sCor2[...,0])
        # # print('dhi0:',self.dHI0[...,0])
        # # print('dhi_pre:',self.dHI_pre[...,0])
        # # print('canopydevend:',self.CanopyDevEndCD[...,0])
        # # print('hiendcd:',self.HIendCD[...,0])
        # # print('flowering:',self.FloweringCD[...,0])
        # # print('ahi:',self.a_HI[...,0])
        # # print('bhi:',self.b_HI[...,0])
        # # print('exc:',self.exc[...,0])
        # # print('y:',self.Y[...,0])
        # # print('b:',self.B[...,0])
        # # print('cropmature:',self.CropMature[...,0])
        # # print('gsday1:',self.GrowingSeasonDayOne[...,0])
        # # print('gddcum:', self.GDDcum[...,0])
        # # print('gsix:',self.GrowingSeasonIndex[...,0])

# class AquaCrop_old(Model):
#     def __init__(self, config, time, domain, init=None):
#         super(AquaCrop, self).__init__(
#             config,
#             time,
#             domain,
#             is_1d=True,
#             init=init
#         )
#         self.weather_module = Weather(self)
#         self.groundwater_module = Groundwater(self)
#         self.crop_module = LandSurface(self)

#     def initial(self):
#         self.weather_module.initial()
#         self.groundwater_module.initial()
#         self.crop_module.initial()

#     def dynamic(self):
#         self.weather_module.dynamic()
#         self.groundwater_module.dynamic()
#         self.crop_module.dynamic()

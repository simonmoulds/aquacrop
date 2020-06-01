#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import warnings

from .io.CarbonDioxide import refconc

import aquacrop_fc

# Crop parameters



class CallFortran(object):
    def __init__(self, model):
        self.model = model

    def initial(self):
        pass

    def dynamic(self):

        layer_ix = self.model.layerIndex + 1
        EvapTimeSteps = 20
        mature = np.int32(self.model.CropMature).copy()        
        
        # print('a',type(self.model.GDD))
        # print(type(self.model.GDDcum)) 
        # print(type(self.model.GDDmethod))
        # print(type(self.model.model.tmax.values)) 
        # print(type(self.model.model.tmin.values))
        # print(type(self.model.Tbase))
        # print(type(self.model.Tupp))
        # print(type(np.int32(self.model.GrowthStage)))
        # print(type(self.model.Canopy10Pct))
        # print(type(self.model.MaxCanopy))
        # print('b',type(self.model.Senescence))
        # print(type(np.int32(self.model.DAP)))
        # print(type(self.model.DelayedCDs))
        # print(type(self.model.DelayedGDDs))
        # print(type(self.model.th))
        # print(type(self.model.th_fc_adj))
        # print(type(np.int32(self.model.WTinSoil)))
        # print(type(self.model.th_sat))
        # print(type(self.model.th_fc))
        # print(type(int(self.model.groundwater.WaterTable)))
        # print('c',type(int(self.model.groundwater.DynamicWaterTable)))
        # print(type(self.model.groundwater.zGW))
        # print(type(self.model.dz))
        # print(type(layer_ix))
        # print(type(self.model.PreIrr))
        # print(type(self.model.IrrMethod))
        # print(type(self.model.Zroot))
        # print(type(self.model.Zmin))
        # print(type(self.model.NetIrrSMT))
        # print(type(self.model.th_wilt))
        # print('d',type(self.model.dz_sum))
        # print(type(self.model.DeepPerc))
        # print(type(self.model.FluxOut))
        # print(type(self.model.k_sat))
        # print(type(self.model.tau))
        # print(type(self.model.Runoff))
        # print(type(self.model.Infl))
        # print(type(self.model.model.prec.values))
        # print(type(np.int32(self.model.DaySubmerged)))
        # print(type(np.int32(self.model.Bunds)))
        # print('e',type(self.model.zBund))
        # print(type(np.int32(self.model.CN)))
        # print(type(np.int32(self.model.adjustCurveNumber)))
        # print(type(self.model.zCN))
        # print(type(self.model.CNbot))
        # print(type(self.model.CNtop))
        # print(type(self.model.thRZ_Act)) 
        # print(type(self.model.thRZ_Sat)) 
        # print(type(self.model.thRZ_Fc)) 
        # print(type(self.model.thRZ_Wp)) 
        # print('f',type(self.model.thRZ_Dry)) 
        # print(type(self.model.thRZ_Aer)) 
        # print(type(self.model.TAW)) 
        # print(type(self.model.Dr)) 
        # print(type(self.model.th_dry)) 
        # print(type(self.model.Aer))
        # print(type(self.model.Irr))      
        # print(type(self.model.IrrCum))   
        # print(type(self.model.IrrNetCum))
        # print(type(self.model.SMT1))
        # print('g',type(self.model.SMT2))
        # print(type(self.model.SMT3))
        # print(type(self.model.SMT4))
        # print(type(self.model.IrrScheduled))
        # print(type(self.model.AppEff))
        # print(type(self.model.model.etref.values))
        # print(type(self.model.MaxIrr))
        # print(type(self.model.IrrInterval))            
        # print(type(self.model.SurfaceStorage))
        # print(type(self.model.CrTot))
        # print('h',type(self.model.aCR))
        # print(type(self.model.bCR))
        # print(type(self.model.fshape_cr))
        # print(type(self.model.dz_layer))
        # print(type(self.model.Germination))
        # print(type(self.model.zGerm))
        # print(type(self.model.GermThr))
        # print(type(self.model.rCor)) 
        # print(type(self.model.Zmax)) 
        # print(type(self.model.PctZmin)) 
        # print('i',type(self.model.Emergence)) 
        # print(type(self.model.MaxRooting)) 
        # print(type(self.model.fshape_r)) 
        # print(type(self.model.fshape_ex)) 
        # print(type(self.model.SxBot))
        # print(type(self.model.SxTop))
        # print(type(self.model.TrRatio))
        # print(type(self.model.zRes))
        # print(type(self.model.CC))
        # print(type(self.model.CCprev))
        # print('j',type(self.model.CCadj))
        # print(type(self.model.CC_NS))
        # print(type(self.model.CCadj_NS))
        # print(type(self.model.CCxW))
        # print(type(self.model.CCxAct))
        # print(type(self.model.CCxW_NS))
        # print(type(self.model.CCxAct_NS))
        # print(type(self.model.CC0adj))
        # print(type(self.model.CCxEarlySen))
        # print(type(self.model.tEarlySen))
        # print('k',type(np.int32(self.model.PrematSenes)))  # not required when all modules use Fortran
        # print(type(self.model.CropDead))
        # print(type(self.model.CC0))
        # print(type(self.model.CCx))
        # print(type(self.model.CGC))
        # print(type(self.model.CDC))
        # print(type(self.model.Maturity))
        # print(type(self.model.CanopyDevEnd))
        # print(type(self.model.ETadj))
        # print(type(self.model.p_up1))
        # print('l',type(self.model.p_up2))
        # print(type(self.model.p_up3))
        # print(type(self.model.p_up4))
        # print(type(self.model.p_lo1))
        # print(type(self.model.p_lo2))
        # print(type(self.model.p_lo3))
        # print(type(self.model.p_lo4))
        # print(type(self.model.fshape_w1))
        # print(type(self.model.fshape_w2))
        # print(type(self.model.fshape_w3))
        # print('m',type(self.model.fshape_w4))
        # print(type(self.model.EsAct))
        # print(type(self.model.Epot))
        # print(type(self.model.WetSurf))
        # print(type(self.model.Wsurf))
        # print(type(self.model.Wstage2))
        # print(type(self.model.EvapZ))
        # print(type(self.model.EvapZmin))
        # print(type(self.model.EvapZmax))
        # print(type(self.model.REW))
        # print('n',type(self.model.Kex))
        # print(type(self.model.CCxW))
        # print(type(self.model.fwcc))
        # print(type(self.model.fevap))
        # print(type(self.model.fWrelExp))
        # print(type(np.int32(self.model.Mulches)))
        # print(type(self.model.fMulch))
        # print(type(self.model.MulchPctGS))
        # print(type(self.model.MulchPctOS))
        # print(type(self.model.model.time.timestep))
        # print('o',type(EvapTimeSteps))
        # print(type(self.model.TrPot0)) 
        # print(type(self.model.TrPot_NS)) 
        # print(type(self.model.TrAct))
        # print(type(self.model.TrAct0)) 
        # print(type(self.model.Tpot))
        # print(type(np.int32(self.model.AerDays))) 
        # print(type(np.int32(self.model.AerDaysComp))) 
        # print(type(np.int32(self.model.AgeDays)))
        # print(type(np.int32(self.model.AgeDays_NS)))
        # print('p',type(np.int32(self.model.DaySubmerged)))
        # print(type(self.model.IrrNet)) 
        # print(type(np.int32(self.model.MaxCanopyCD))) 
        # print(type(self.model.Kcb)) 
        # print(type(self.model.CCxW_NS))
        # print(type(self.model.a_Tr))
        # print(type(self.model.fage)) 
        # print(type(np.int32(self.model.LagAer))) 
        # print(type(self.model.CurrentConc)) 
        # print(type(refconc))
        # print('q',type(self.model.ETpot))
        # print(type(self.model.GwIn))
        # print(type(self.model.HI)) 
        # print(type(self.model.PctLagPhase))
        # print(type(self.model.YieldForm))
        # print(type(self.model.CCmin)) 
        # print(type(self.model.HIini)) 
        # print(type(self.model.HI0)) 
        # print(type(self.model.HIGC)) 
        # print(type(self.model.HIstart)) 
        # print('r',type(self.model.HIstartCD)) 
        # print(type(self.model.tLinSwitch)) 
        # print(type(self.model.dHILinear)) 
        # print(type(self.model.CropType))
        # print(type(self.model.BioTempStress))
        # print(type(self.model.GDD_up))
        # print(type(self.model.GDD_lo))
        # print(type(self.model.PolHeatStress))
        # print(type(self.model.Tmax_up))
        # print(type(self.model.Tmax_lo))
        # print('s',type(self.model.fshape_b))
        # print(type(self.model.PolColdStress))
        # print(type(self.model.Tmin_up))
        # print(type(self.model.Tmin_lo))
        # print(type(self.model.B))
        # print(type(self.model.B_NS))
        # print(type(self.model.YldFormCD))
        # print(type(self.model.WP))
        # print(type(self.model.WPy))
        # print(type(self.model.fCO2))
        # print('t',type(self.model.Determinant))
        # print(type(self.model.HIadj))
        # print(type(self.model.PreAdj))
        # print(type(self.model.Fpre))
        # print(type(self.model.Fpol)) 
        # print(type(self.model.Fpost))
        # print(type(self.model.fpost_dwn))
        # print(type(self.model.fpost_upp))
        # print(type(self.model.sCor1))
        # print(type(self.model.sCor2))
        # print('u',type(self.model.dHI0)) 
        # print(type(self.model.dHI_pre)) 
        # print(type(self.model.CanopyDevEndCD)) 
        # print(type(self.model.HIendCD)) 
        # print(type(self.model.FloweringCD)) 
        # print(type(self.model.a_HI)) 
        # print(type(self.model.b_HI)) 
        # print(type(self.model.exc))
        # print(type(self.model.Y))
        # print(type(mature))
        
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
            self.model.HIadj.T,
            self.model.PreAdj.T,
            self.model.Fpre.T, 
            self.model.Fpol.T, 
            self.model.Fpost.T, 
            self.model.fpost_dwn.T, 
            self.model.fpost_upp.T, 
            self.model.sCor1.T, 
            self.model.sCor2.T,
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

# self.model.GDD.T, 
# self.model.GDDcum.T, 
# self.model.GDDmethod, 
# self.model.model.tmax.values.T, 
# self.model.model.tmin.values.T,
# self.model.Tbase.T,
# self.model.Tupp.T,
# np.int32(self.model.GrowthStage).T,
# self.model.Canopy10Pct.T,
# self.model.MaxCanopy.T,
# self.model.Senescence.T,
# np.int32(self.model.DAP).T,
# self.model.DelayedCDs.T,
# self.model.DelayedGDDs.T,
# self.model.th.T,
# self.model.th_fc_adj.T,
# np.int32(self.model.WTinSoil).T,
# self.model.th_sat.T,
# self.model.th_fc.T,
# int(self.model.groundwater.WaterTable),
# int(self.model.groundwater.DynamicWaterTable),
# self.model.groundwater.zGW,
# self.model.dz,
# layer_ix,
# self.model.PreIrr.T,
# self.model.IrrMethod.T,
# self.model.Zroot.T,
# self.model.Zmin.T,
# self.model.NetIrrSMT.T,
# self.model.th_wilt.T,
# self.model.dz_sum,
# self.model.DeepPerc.T,
# self.model.FluxOut.T,
# self.model.k_sat.T,
# self.model.tau.T,
# self.model.Runoff.T,
# self.model.Infl.T,
# self.model.model.prec.values.T,
# np.int32(self.model.DaySubmerged).T,
# np.int32(self.model.Bunds).T,
# self.model.zBund.T,
# np.int32(self.model.CN).T,
# np.int32(self.model.adjustCurveNumber),
# self.model.zCN.T,
# self.model.CNbot.T,
# self.model.CNtop.T,
# self.model.thRZ_Act.T, 
# self.model.thRZ_Sat.T, 
# self.model.thRZ_Fc.T, 
# self.model.thRZ_Wp.T, 
# self.model.thRZ_Dry.T, 
# self.model.thRZ_Aer.T, 
# self.model.TAW.T, 
# self.model.Dr.T, 
# self.model.th_dry.T, 
# self.model.Aer.T,
# self.model.Irr.T,      
# self.model.IrrCum.T,   
# self.model.IrrNetCum.T,
# self.model.SMT1.T,
# self.model.SMT2.T,
# self.model.SMT3.T,
# self.model.SMT4.T,
# self.model.IrrScheduled.T,  # TODO
# self.model.AppEff.T,
# self.model.model.etref.values.T,
# self.model.MaxIrr.T,
# self.model.IrrInterval.T,            
# self.model.SurfaceStorage.T,
# self.model.CrTot.T,
# self.model.aCR.T,
# self.model.bCR.T,
# self.model.fshape_cr.T,
# self.model.dz_layer,
# self.model.Germination.T,
# self.model.zGerm.T,
# self.model.GermThr.T,
# self.model.rCor.T, 
# self.model.Zmax.T, 
# self.model.PctZmin.T, 
# self.model.Emergence.T, 
# self.model.MaxRooting.T, 
# self.model.fshape_r.T, 
# self.model.fshape_ex.T, 
# self.model.SxBot.T,
# self.model.SxTop.T,
# self.model.TrRatio.T,
# self.model.zRes.T,
# self.model.CC.T,
# self.model.CCprev.T,
# self.model.CCadj.T,
# self.model.CC_NS.T,
# self.model.CCadj_NS.T,
# self.model.CCxW.T,
# self.model.CCxAct.T,
# self.model.CCxW_NS.T,
# self.model.CCxAct_NS.T,
# self.model.CC0adj.T,
# self.model.CCxEarlySen.T,
# self.model.tEarlySen.T,
# np.int32(self.model.PrematSenes).T,  # not required when all modules use Fortran
# self.model.CropDead.T,
# self.model.CC0.T,
# self.model.CCx.T,
# self.model.CGC.T,
# self.model.CDC.T,
# self.model.Maturity.T,
# self.model.CanopyDevEnd.T,
# self.model.ETadj.T,
# self.model.p_up1.T,
# self.model.p_up2.T,
# self.model.p_up3.T,
# self.model.p_up4.T,
# self.model.p_lo1.T,
# self.model.p_lo2.T,
# self.model.p_lo3.T,
# self.model.p_lo4.T,
# self.model.fshape_w1.T,
# self.model.fshape_w2.T,
# self.model.fshape_w3.T,
# self.model.fshape_w4.T,
# self.model.EsAct.T,
# self.model.Epot.T,
# self.model.WetSurf.T,
# self.model.Wsurf.T,
# self.model.Wstage2.T,
# self.model.EvapZ.T,
# self.model.EvapZmin.T,
# self.model.EvapZmax.T,
# self.model.REW.T,
# self.model.Kex.T,
# self.model.CCxW.T,
# self.model.fwcc.T,
# self.model.fevap.T,
# self.model.fWrelExp.T,
# np.int32(self.model.Mulches).T,
# self.model.fMulch.T,
# self.model.MulchPctGS.T,
# self.model.MulchPctOS.T,
# self.model.model.time.timestep,
# EvapTimeSteps,
# self.model.TrPot0.T, 
# self.model.TrPot_NS.T, 
# self.model.TrAct.T,
# self.model.TrAct0.T, 
# self.model.Tpot.T, 
# np.int32(self.model.AerDays).T, 
# np.int32(self.model.AerDaysComp).T, 
# np.int32(self.model.AgeDays).T,
# np.int32(self.model.AgeDays_NS).T,
# np.int32(self.model.DaySubmerged).T,
# self.model.IrrNet.T, 
# np.int32(self.model.MaxCanopyCD).T, 
# self.model.Kcb.T, 
# self.model.CCxW_NS.T,
# self.model.a_Tr.T,
# self.model.fage.T, 
# np.int32(self.model.LagAer).T, 
# self.model.CurrentConc.T, 
# refconc,
# self.model.ETpot.T,
# self.model.GwIn.T,
# self.model.HI.T, 
# self.model.PctLagPhase.T,
# self.model.YieldForm.T,
# self.model.CCmin.T, 
# self.model.HIini.T, 
# self.model.HI0.T, 
# self.model.HIGC.T, 
# self.model.HIstart.T, 
# self.model.HIstartCD.T, 
# self.model.tLinSwitch.T, 
# self.model.dHILinear.T, 
# self.model.CropType.T,
# self.model.BioTempStress.T,
# self.model.GDD_up.T,
# self.model.GDD_lo.T,
# self.model.PolHeatStress.T,
# self.model.Tmax_up.T,
# self.model.Tmax_lo.T,
# self.model.fshape_b.T,
# self.model.PolColdStress.T,
# self.model.Tmin_up.T,
# self.model.Tmin_lo.T,
# self.model.B.T,
# self.model.B_NS.T,
# self.model.YldFormCD.T,
# self.model.WP.T,
# self.model.WPy.T,
# self.model.fCO2.T,
# self.model.Determinant.T,
# np.float64(self.model.HIadj).T,
# self.model.PreAdj.T,
# np.float64(self.model.Fpre).T, 
# np.float64(self.model.Fpol).T, 
# np.float64(self.model.Fpost).T, 
# np.float64(self.model.fpost_dwn).T, 
# np.float64(self.model.fpost_upp).T, 
# np.float64(self.model.sCor1).T, 
# np.float64(self.model.sCor2).T,
# self.model.dHI0.T, 
# self.model.dHI_pre.T, 
# self.model.CanopyDevEndCD.T, 
# self.model.HIendCD.T, 
# self.model.FloweringCD.T, 
# self.model.a_HI.T, 
# self.model.b_HI.T, 
# self.model.exc.T,
# self.model.Y.T,
# mature.T,
# int(self.model.CalendarType), self.model.GrowingSeasonDayOne.T, self.model.GrowingSeasonIndex.T,
# self.model.nFarm, self.model.nCrop, self.model.nComp, self.model.nLayer, self.model.domain.nxy            
        

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numba import jit

class CanopyCover(object):
    """Class to represent canopy growth/decline."""
    def __init__(self, CanopyCover_variable):
        """Create a CanopyCover object by providing an AquaCrop 
        object.
        """
        self.var = CanopyCover_variable

    def initial(self):
        """Initialize the NumPy data structures modified by the class 
        instance.
        """
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.tEarlySen = np.copy(arr_zeros)
        self.var.CC = np.copy(arr_zeros)
        self.var.CCadj = np.copy(arr_zeros)
        self.var.CC_NS = np.copy(arr_zeros)
        self.var.CCadj_NS = np.copy(arr_zeros)
        self.var.CCxAct = np.copy(arr_zeros)
        self.var.CCxAct_NS = np.copy(arr_zeros)
        self.var.CCxW = np.copy(arr_zeros)
        self.var.CCxW_NS = np.copy(arr_zeros)
        self.var.CCxEarlySen = np.copy(arr_zeros)
        self.var.CCprev = np.copy(arr_zeros)
        self.var.PrematSenes = np.copy(arr_zeros.astype(bool))        
        self.var.CropDead = np.copy(arr_zeros.astype(bool))        
        self.var.CC0adj = np.copy(arr_zeros)

    def reset_initial_conditions(self):
        """Reset the value of various variables at the beginning of 
        the growing season.
        """
        self.var.tEarlySen[self.var.GrowingSeasonDayOne] = 0
        self.var.CC[self.var.GrowingSeasonDayOne] = 0
        self.var.CCadj[self.var.GrowingSeasonDayOne] = 0
        self.var.CC_NS[self.var.GrowingSeasonDayOne] = 0
        self.var.CCadj_NS[self.var.GrowingSeasonDayOne] = 0
        self.var.CCxAct[self.var.GrowingSeasonDayOne] = 0
        self.var.CCxAct_NS[self.var.GrowingSeasonDayOne] = 0
        self.var.CCxW[self.var.GrowingSeasonDayOne] = 0
        self.var.CCxW_NS[self.var.GrowingSeasonDayOne] = 0
        self.var.CCxEarlySen[self.var.GrowingSeasonDayOne] = 0
        self.var.CCprev[self.var.GrowingSeasonDayOne] = 0
        self.var.PrematSenes[self.var.GrowingSeasonDayOne] = False
        self.var.CropDead[self.var.GrowingSeasonDayOne] = False
        self.var.CC0adj[self.var.GrowingSeasonDayOne] = self.var.CC0[self.var.GrowingSeasonDayOne]

    @staticmethod
    # @jit(nopython=True, cache=True)
    def canopy_cover_development_exponential_growth(CC0, CGC, dt):
        CC = (CC0 * np.exp(CGC * dt))
        return CC

    @staticmethod
    # @jit(nopython=False, cache=True)
    def canopy_cover_development_exponential_decline(CC0, CCx, CGC, dt):
        # Numba doesn't like np.divide - I think it doesn't support the where argument
        CC = (CCx - 0.25 * np.divide(CCx, CC0, out=np.zeros_like(CC0), where=CC0!=0) * CCx * np.exp(-CGC * dt))
        return CC
    
    def canopy_cover_development(self, CC0, CCx, CGC, dt):
        """Function to compute canopy development during the 
        exponential growth phase. This is detailed in Section
        3.4.2 of the FAO AquaCrop manual.
        """
        CC = self.canopy_cover_development_exponential_growth(CC0, CGC, dt)
        CC_decline = self.canopy_cover_development_exponential_decline(CC0, CCx, CGC, dt)
        declining_phase = CC > (CCx / 2.)
        CC[declining_phase] = CC_decline[declining_phase]
        CC = np.clip(CC, 0, CCx)
        return CC

    def green_canopy_cover_decline(self, CCx, CDC, dt):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        CC = np.copy(arr_zeros)
        cond2 = (CCx >= 0.001)
        CC[cond2] = (CCx * (1. - 0.05 * (np.exp(dt * np.divide(CDC, CCx, out=np.copy(arr_zeros), where=CCx!=0)) - 1.)))[cond2]
        CC = np.clip(CC, 0, 1)
        return CC

    @staticmethod
    def compute_target_cgc(x, tSum, dt):
        """Function to compute factor to adjust current CGC 
        to reach CC on previous day.
        """
        CGCx_divd = np.log(x, out=np.zeros_like(x), where=x>0)
        CGCx_divs = tSum - dt
        CGCx = np.divide(CGCx_divd, CGCx_divs, out=np.zeros_like(x), where=CGCx_divs!=0)
        return CGCx
        
    def canopy_cover_required_time_cgc_mode(self, CC0, CCx, CGC, dt, tSum):
        """Function to find time required to reach canopy 
        cover at end of previous day, given current CGC.
        """
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        CGCx = np.copy(arr_zeros)
        cond1 = (self.var.CCprev <= (CCx / 2))
        x = np.divide(self.var.CCprev, CC0, out=np.copy(arr_zeros), where=CC0!=0)
        CGCx[cond1] = self.compute_target_cgc(x, tSum, dt)[cond1]
        cond2 = np.logical_not(cond1)
        x1 = np.divide(0.25 * CCx * CCx, CC0, out=np.copy(arr_zeros), where=CC0!=0)
        x2 = CCx - self.var.CCprev
        x3 = np.divide(x1, x2, out=np.copy(arr_zeros), where=x2!=0)
        CGCx[cond2] = self.compute_target_cgc(x3, tSum, dt)[cond2]
        tReq = (tSum - dt) * np.divide(CGCx, CGC, out=np.copy(arr_zeros), where=CGC!=0)
        return tReq
    
    def canopy_cover_required_time_cdc_mode(self, CCx, CDC):
        """Function to find time required to reach canopy 
        cover at end of previous day, given current CDC.
        """
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        x1 = np.divide(self.var.CCprev, CCx, out=np.copy(arr_zeros), where=CCx!=0)
        x1 = np.clip(x1,0,1.)  # try this
        x2 = 1. + (1. - x1) / 0.05  # should CCprev ever be greater than CCx???            
        tReq_divd = np.log(x2, out=np.copy(arr_zeros), where=x2!=0)  # TODO: sort this out!!!
        tReq_divs = np.divide(CDC, CCx, out=np.copy(arr_zeros), where=CCx!=0)
        tReq = np.divide(tReq_divd, tReq_divs, out=np.copy(arr_zeros), where=tReq_divs!=0)
        return tReq
            
    def adjust_CCx(self, CC0, CCx, CGC, CDC, dt, tSum, CanopyDevEnd):
        """Adjust CCx value for changes in CGC due to water 
        stress during the growing season.
        """
        tCCtmp = self.canopy_cover_required_time_cgc_mode(CC0, CCx, CGC, dt, tSum)
        cond1 = (tCCtmp > 0)
        tCCtmp[cond1] += ((CanopyDevEnd - tSum) + dt)[cond1]
        CCxAdj = self.canopy_cover_development(CC0, CCx, CGC, tCCtmp)
        CCxAdj[np.logical_not(cond1)] = 0
        return CCxAdj

    def adjust_CGC_for_water_stress(self):
        return self.var.CGC * self.var.Ksw_Exp
    
    def adjust_CDC_for_water_stress(self):
        CDCadj = ((1. - (self.var.Ksw_Sen ** 8)) * self.var.CDC)
        CDCadj[self.var.Ksw_Sen > 0.99999] = 0.0001
        return CDCadj

    def adjust_CDC_late_stage(self, CDC, CCx, CCxAct):
        return CDC * np.divide(CCxAct, CCx, out=np.zeros_like(CCx), where=CCx!=0)

    def is_initial_stage(self, CC0, CCprev, tCC):
        initial_stage = (
            self.var.GrowingSeasonIndex
            & (tCC >= self.var.Emergence)
            & (tCC < self.var.CanopyDevEnd)
            & (CCprev <= CC0)
        )
        return initial_stage
    
    def is_crop_development_stage(self, CC0, CCprev, tCC):
        crop_development_stage = (
            self.var.GrowingSeasonIndex
            & (tCC >= self.var.Emergence)
            & (tCC < self.var.CanopyDevEnd)
            & (CCprev > CC0)
        )
        return crop_development_stage
    
    def is_canopy_approaching_maximum_size(self, crop_development_stage, CCprev, CCx):
        return crop_development_stage & (CCprev >= (0.9799 * CCx))
    
    def is_mid_stage(self, tCC):
        mid_stage = (
            self.var.GrowingSeasonIndex
            & (tCC >= self.var.Emergence)
            & (tCC >= self.var.CanopyDevEnd)
            & (tCC < self.var.Senescence)
        )
        return mid_stage

    def is_late_stage(self, tCC):
        late_stage = (
            self.var.GrowingSeasonIndex
            & (tCC >= self.var.Emergence)
            & (tCC >= self.var.CanopyDevEnd)
            & (tCC >= self.var.Senescence)
        )
        return late_stage
                
    def potential_canopy_development(self, tCC, dtCC):
        """Compute potential canopy development."""

        CC_NSprev = np.copy(self.var.CC_NS)
        self.var.CC_NS.fill(0)
        
        CC_NS_initial = self.canopy_cover_development(self.var.CC0, self.var.CCx, self.var.CGC, dtCC)
        initial_stage = self.is_initial_stage(self.var.CC0, CC_NSprev, tCC)
        self.var.CC_NS[initial_stage] = CC_NS_initial[initial_stage]

        CC_NS_development = self.canopy_cover_development(self.var.CC0, self.var.CCx, self.var.CGC, tCC - self.var.Emergence)
        crop_development_stage = self.is_crop_development_stage(self.var.CC0, CC_NSprev, tCC)
        self.var.CC_NS[crop_development_stage] = CC_NS_development[crop_development_stage]

        # Update maximum canopy cover size in growing season
        self.var.CCxAct_NS[(initial_stage | crop_development_stage)] = self.var.CC_NS[(initial_stage | crop_development_stage)]

        # Mid-season stage - no canopy growth, so do not update CC_NS
        mid_stage = self.is_mid_stage(tCC)
        self.var.CC_NS[mid_stage] = CC_NSprev[mid_stage]
        self.var.CCxAct_NS[mid_stage] = self.var.CC_NS[mid_stage]
        
        # Late-season stage - canopy decline
        late_stage = self.is_late_stage(tCC)
        CC_NS_decline = self.green_canopy_cover_decline(self.var.CCx, self.var.CDC, tCC - self.var.Emergence)
        self.var.CC_NS[late_stage] = CC_NS_decline[late_stage]

        # Set CCx for calculation of withered canopy effects
        self.var.CCxW_NS[(mid_stage | late_stage)] = self.var.CCxAct_NS[(mid_stage | late_stage)]
        
    def actual_canopy_development(self, tCC, tCCadj, dtCC):
        """Compute actual canopy development."""

        # No canopy development before emergence/germination or after maturity
        cond4 = (self.var.GrowingSeasonIndex & ((tCCadj < self.var.Emergence) | (np.round(tCCadj) > self.var.Maturity)))
        self.var.CC[cond4] = 0
        self.var.CC.fill(0)
        
        # Very small initial CC as it is first day or due to senescence. In
        # this case, assume no leaf expansion stress
        CC_initial = self.canopy_cover_development(self.var.CC0adj, self.var.CCx, self.var.CGC, dtCC)
        initial_stage = self.is_initial_stage(self.var.CC0adj, self.var.CCprev, tCCadj)
        self.var.CC[initial_stage] = CC_initial[initial_stage]
        
        # Otherwise, canopy growth can occur
        cond5 = (self.var.GrowingSeasonIndex & np.logical_not(cond4) & (tCCadj < self.var.CanopyDevEnd))
        crop_development_stage = self.is_crop_development_stage(self.var.CC0, self.var.CCprev, tCCadj)
        self.var.CC[crop_development_stage] = self.var.CCprev[crop_development_stage] # **TEST**
        
        canopy_approaching_max_size = self.is_canopy_approaching_maximum_size(crop_development_stage, self.var.CCprev, self.var.CCx)
        CC_max = self.canopy_cover_development(self.var.CC0, self.var.CCx, self.var.CGC, tCC - self.var.Emergence)
        self.var.CC[canopy_approaching_max_size] = CC_max[canopy_approaching_max_size]

        # adjust canopy growth coefficient for the effects of
        # leaf expansion under water stress
        CGCadj = self.adjust_CGC_for_water_stress()
        CCxAdj = self.adjust_CCx(self.var.CC0adj, self.var.CCx, CGCadj, self.var.CDC, dtCC, tCCadj, self.var.CanopyDevEnd)
        canopy_growth_after_adjustment = (
            np.logical_not(canopy_approaching_max_size)
            & (CGCadj > 0)
            & (CCxAdj > 0)
        )            
        canopy_approaching_max_size_adj = (
            canopy_growth_after_adjustment
            & (np.abs(self.var.CCprev - self.var.CCx) < 0.00001)
        )                
        # Approaching maximum canopy size
        CC_max_adj = self.canopy_cover_development(self.var.CC0, self.var.CCx, self.var.CGC, tCC - self.var.Emergence)
        self.var.CC[canopy_approaching_max_size_adj] = CC_max_adj[canopy_approaching_max_size_adj]

        # Determine time required to reach CC on previous day, given CGCadj
        # value
        tReq = self.canopy_cover_required_time_cgc_mode(self.var.CC0adj, CCxAdj, CGCadj, dtCC, tCCadj)
        tmp_tCC = tReq + dtCC
        
        cond522112 = (
            canopy_growth_after_adjustment
            & np.logical_not(canopy_approaching_max_size_adj)
        )
        
        # Determine new canopy size
        cond5221121 = (cond522112 & (tmp_tCC > 0))
        CC_growth = self.canopy_cover_development(self.var.CC0adj, CCxAdj, CGCadj, tmp_tCC)
        self.var.CC[cond5221121] = CC_growth[cond5221121]

        canopy_growth_after_adjustment = (
            np.logical_not(canopy_approaching_max_size)
            & (CGCadj > 0)
            & (CCxAdj > 0)
        )
        
        # No canopy growth (line 119)
        cond5222 = (crop_development_stage & np.logical_not(canopy_approaching_max_size) & np.logical_not(CGCadj > 0))
        cond52221 = cond5222 & (self.var.CC < self.var.CC0adj)
        self.var.CC0adj[cond52221] = self.var.CC[cond52221]
        
        # No more canopy growth is possible or canopy is in decline (line 132)
        cond6 = (self.var.GrowingSeasonIndex & np.logical_not(cond4 | cond5) & (tCCadj > self.var.CanopyDevEnd))

        # Mid-season stage - no canopy growth: update actual maximum canopy
        # cover size during growing season only (i.e. do not update CC)
        mid_stage = self.is_mid_stage(tCCadj)
        self.var.CC[mid_stage] = self.var.CCprev[mid_stage]
        cond53 = (self.var.GrowingSeasonIndex & (self.var.CC > self.var.CCxAct))
        self.var.CCxAct[cond53] = self.var.CC[cond53]
        
        cond61 = (cond6 & (tCCadj < self.var.Senescence))

        # Late season stage - canopy decline: update canopy decline coefficient
        # for difference between actual and potential CCx, and determine new
        # canopy size
        late_stage = self.is_late_stage(tCCadj)
        CDCadj = self.adjust_CDC_late_stage(self.var.CDC, self.var.CCx, self.var.CCxAct)
        CC_decline = self.green_canopy_cover_decline(self.var.CCxAct, CDCadj, tCCadj - self.var.Senescence)
        self.var.CC[late_stage] = CC_decline[late_stage]
        
    def adjust_CCx_for_late_stage_rewatering(self, dt):
        CCxAdj = self.var.CCprev / (1. - 0.05 * (np.exp(dt * (np.divide(self.var.CDC, self.var.CCx, out=np.zeros_like(self.var.CCx), where=self.var.CCx!=0))) - 1.))
        return CCxAdj

    def adjust_CDC_for_late_stage_rewatering(self, dt, CCxAdj):
        CDCadj = self.var.CDC * np.divide(CCxAdj, self.var.CCx, out=np.zeros_like(self.var.CCx), where=self.var.CCx!=0)
        return CDCadj
        
    def compute_canopy_senescence_due_to_water_stress(self, tCCadj):
        pass

    def time_step(self):        
        if self.var.CalendarType == 1:
            dtCC = np.ones((self.var.nFarm, self.var.nCrop, self.var.nCell))
        elif self.var.CalendarType == 2:
            dtCC = np.copy(self.var.GDD)
        return dtCC
    
    def time_since_planting(self):
        if self.var.CalendarType == 1:
            tCC = np.copy(self.var.DAP)
        elif self.var.CalendarType == 2:
            tCC = np.copy(self.var.GDDcum)
        return tCC
    
    def adjust_time_since_planting(self):
        if self.var.CalendarType == 1:
            tCCadj = self.var.DAP - self.var.DelayedCDs
        elif self.var.CalendarType == 2:
            tCCadj = self.var.GDDcum - self.var.DelayedGDDs
        return tCCadj

    def is_crop_dead(self, tCCadj):
        cond6 = (
            self.var.GrowingSeasonIndex
            & ((tCCadj <= self.var.Maturity)
               & (tCCadj >= self.var.Emergence)
               & (tCCadj > self.var.CanopyDevEnd))
        )
        cond63 = (cond6 & ((self.var.CC < 0.001) & np.logical_not(self.var.CropDead)))
        return cond63

    def canopy_cover_after_senescence(self, tCCadj, dtCC):        
        CDCadj = self.adjust_CDC_for_water_stress()        
        CCsen = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        cond7115 = self.var.CCxEarlySen >= 0.001
        tReq = self.canopy_cover_required_time_cdc_mode(self.var.CCxEarlySen, CDCadj)
        CCsen_decline = self.green_canopy_cover_decline(self.var.CCxEarlySen, CDCadj, tReq + dtCC)
        CCsen[cond7115] = CCsen_decline[cond7115]
        # print('CCsen:',CCsen[0])
        return CCsen

    def adjust_canopy_cover_after_senescence(self, tCCadj, dtCC):
        CCsen = self.canopy_cover_after_senescence(tCCadj, dtCC)
        cond7116 = (self.early_canopy_senescence & (tCCadj < self.var.Senescence))
        CCsen[cond7116] = np.clip(CCsen, None, self.var.CCx)[cond7116]
        self.var.CC[cond7116] = CCsen[cond7116]        
        self.var.CC[cond7116] = np.clip(self.var.CC, None, self.var.CCprev)[cond7116]
        self.var.CC[cond7116] = np.clip(self.var.CC, None, self.var.CCx)[cond7116]  # I think this is unnecessary
        self.var.CCxAct[cond7116] = self.var.CC[cond7116]

        # def adjust_CC0_after_senescence(self):
        # Update CC0 if current CC is less than initial canopy cover size at
        # planting
        cond71161 = (cond7116 & (self.var.CC < self.var.CC0))
        self.var.CC0adj[cond71161] = self.var.CC[cond71161]
        cond71162 = (cond7116 & np.logical_not(cond71161))
        self.var.CC0adj[cond71162] = self.var.CC0[cond71162]

        # Update CC to account for canopy cover senescence due to water stress
        cond7117 = (self.early_canopy_senescence & np.logical_not(cond7116))
        self.var.CC[cond7117] = np.clip(self.var.CC, None, CCsen)[cond7117]

        # Check for crop growth termination
        cond7118 = (self.early_canopy_senescence & ((self.var.CC < 0.001) & np.logical_not(self.var.CropDead)))
        self.var.CC[cond7118] = 0
        self.var.CropDead[cond7118] = True

    def update_canopy_cover_after_rewatering_in_late_season(self, tCCadj, dtCC):

        cond712 = (self.model_effects_of_canopy_senescence & np.logical_not(self.early_canopy_senescence))
        self.var.PrematSenes[cond712] = False  # TODO: I don't think this is needed
        
        # def update_CC_after_rewatering_in_late_season(self, tCCadj, dtCC):
        # Rewatering of canopy in late season: get adjusted values of CCx and
        # CDC and update CC
        cond7121 = (cond712 & ((tCCadj > self.var.Senescence) & (self.var.tEarlySen > 0)))
        tmp_tCC = tCCadj - dtCC - self.var.Senescence
        CCxAdj = self.adjust_CCx_for_late_stage_rewatering(tmp_tCC)
        CDCadj = self.adjust_CDC_for_late_stage_rewatering(tmp_tCC, CCxAdj)
        tmp_tCC = tCCadj - self.var.Senescence
        tmp_CC = self.green_canopy_cover_decline(CCxAdj, CDCadj, tmp_tCC)
        self.var.CC[cond7121] = tmp_CC[cond7121]

        # Check for crop growth termination
        cond71211 = (cond7121 & ((self.var.CC < 0.001) & np.logical_not(self.var.CropDead)))
        self.var.CC[cond71211] = 0
        self.var.CropDead[cond71211] = True

        # Reset early senescence counter
        self.var.tEarlySen[cond712] = 0
        
    def canopy_senescence_due_to_water_stress(self, tCCadj, dtCC):
            
        # Check for early canopy senescence starting/continuing due to severe
        # water stress
        self.model_effects_of_canopy_senescence = (
            self.var.GrowingSeasonIndex
            & (tCCadj >= self.var.Emergence)
            & ((tCCadj < self.var.Senescence)
               | (self.var.tEarlySen > 0))
        )
        
        # Early canopy senescence
        self.early_canopy_senescence = self.model_effects_of_canopy_senescence & (self.var.Ksw_Sen < 1.)
        self.var.PrematSenes[self.early_canopy_senescence] = True
        self.var.tEarlySen[self.early_canopy_senescence] += dtCC[self.early_canopy_senescence]
        
        # No prior early senescence
        no_prior_early_senescence = (self.early_canopy_senescence & (self.var.tEarlySen == 0))
        self.var.CCxEarlySen[no_prior_early_senescence] = self.var.CCprev[no_prior_early_senescence]

        # run water stress module
        self.var.water_stress_module.dynamic(beta=False)
        self.adjust_canopy_cover_after_senescence(tCCadj, dtCC)
        self.update_canopy_cover_after_rewatering_in_late_season(tCCadj, dtCC)

        # Adjust CCx for effects of withered canopy
        self.var.CCxW[self.model_effects_of_canopy_senescence] = np.clip(self.var.CCxW, self.var.CC, None)[self.model_effects_of_canopy_senescence]

    def adjust_canopy_cover_for_microadvective_effects(self, tCC):
        # Check to ensure potential CC is not slightly lower than actual
        cond8 = (self.var.GrowingSeasonIndex & (self.var.CC_NS < self.var.CC))
        self.var.CC_NS[cond8] = self.var.CC[cond8]
        cond81 = (cond8 & (tCC < self.var.CanopyDevEnd))
        self.var.CCxAct_NS[cond81] = self.var.CC_NS[cond81]
        # Actual (with water stress)
        self.var.CCadj[self.var.GrowingSeasonIndex] = ((1.72 * self.var.CC) - (self.var.CC ** 2) + (0.3 * (self.var.CC ** 3)))[self.var.GrowingSeasonIndex]
        # Potential (without water stress)
        self.var.CCadj_NS[self.var.GrowingSeasonIndex] = ((1.72 * self.var.CC_NS) - (self.var.CC_NS ** 2) + (0.3 * (self.var.CC_NS ** 3)))[self.var.GrowingSeasonIndex]
        
    def dynamic(self):
        """Update CanopyCover object for the current time step."""
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()            

        # Store initial condition
        self.var.CCprev = np.copy(self.var.CC)

        # get some time variables
        dtCC = self.time_step()
        tCC = self.time_since_planting()
        tCCadj = self.adjust_time_since_planting()
        # Canopy development (potential)
        self.potential_canopy_development(tCC, dtCC)
        # Canopy development (actual)
        self.actual_canopy_development(tCC, tCCadj, dtCC)
        crop_dead = self.is_crop_dead(tCCadj)
        self.var.CC[crop_dead] = 0
        self.var.CropDead[crop_dead] = True

        # Canopy senescence due to water stress (actual)
        self.canopy_senescence_due_to_water_stress(tCCadj, dtCC)
        
        # adjust for micro-advective effects
        self.adjust_canopy_cover_for_microadvective_effects(tCC)        

        self.var.CC[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        self.var.CCadj[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        self.var.CC_NS[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        self.var.CCadj_NS[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        self.var.CCxW[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        self.var.CCxAct[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        self.var.CCxW_NS[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        self.var.CCxAct_NS[np.logical_not(self.var.GrowingSeasonIndex)] = 0

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class HarvestIndex(object):
    """Class to represent harvest index on current day."""
    
    def __init__(self, model_object):
        self.var = model_object

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.YieldForm = np.copy(arr_zeros)
        self.var.HI = np.copy(arr_zeros)
        self.var.HIt = np.copy(arr_zeros)
        self.var.PctLagPhase = np.copy(arr_zeros)

    def reset_initial_conditions(self):
        self.var.PctLagPhase[self.var.GrowingSeasonDayOne] = 0
        self.var.HI[self.var.GrowingSeasonDayOne] = 0

    def compute_harvest_index_with_logistic_growth(self, time):
        """Compute increase in harvest index.

        This function applies a logistic function (AquaCrop 
        manual Ch3, Eqn 3.12b) to compute the increase in 
        harvest index for the current day.
        """
        HI_divd = (self.var.HIini * self.var.HI0)
        HI_divs = (self.var.HIini + (self.var.HI0 - self.var.HIini) * np.exp(-self.var.HIGC * time))#self.var.HIt))
        HI = np.divide(HI_divd, HI_divs, out=np.zeros_like(HI_divs), where=HI_divs!=0)
        return HI

    def is_yield_formation_period(self):
        """Find out whether a crop is in its yield formation phase.

        Yield formation begins when the time since germination exceeds
        crop parameter HIstart.
        """
        self.var.YieldForm = (
            self.var.GrowingSeasonIndex
            & (self.var.time_since_germination > self.var.HIstart)
        )

    def compute_harvest_index_time(self):
        self.var.HIt = self.var.DAP - self.var.DelayedCDs - self.var.HIstartCD - 1.

    def compute_harvest_index_crop_type_1_or_2(self):
        """Compute harvest index for crops belonging to types 1 or 2.

        These crops are leafy vegetables (type 1) or root/tubers (type 2). 
        In this case, the build-up of harvest-index is entirely governed 
        by a logistic function.
        """
        use_logistic_growth_crop_type_1_or_2 = (
            self.compute_harvest_index
            & ((self.var.CropType == 1) | (self.var.CropType == 2)))
        
        self.var.PctLagPhase[use_logistic_growth_crop_type_1_or_2] = 100
        
        HI = self.compute_harvest_index_with_logistic_growth(self.var.HIt)
        self.var.HI[use_logistic_growth_crop_type_1_or_2] = HI[use_logistic_growth_crop_type_1_or_2]

        harvest_index_approaching_maximum = (
            use_logistic_growth_crop_type_1_or_2
            & (self.var.HI >= (0.9799 * self.var.HI0))
        )
        self.var.HI[harvest_index_approaching_maximum] = self.var.HI0[harvest_index_approaching_maximum]

    def compute_harvest_index_crop_type_3(self):
        """Compute harvest index for crops belonging to type 3.

        Crops type 3 represents fruits/grains. The function
        which governs the build-up of these crops consists of 
        a logistic part and and a linear part, determined by the
        crop parameter tLinSwitch. When time (HIt) is less than
        tLinSwitch the build-up is logistic; thereafter it is
        linear. See AquaCrop manual, Ch3, section 3.12.2.
        """
        use_logistic_growth_crop_type_3 = (
            self.compute_harvest_index
            & ((self.var.CropType == 3)
               & (self.var.HIt < self.var.tLinSwitch)))

        use_linear_growth = (
            self.compute_harvest_index
            & ((self.var.CropType == 3)
               & (self.var.HIt >= self.var.tLinSwitch)))
        compute_harvest_index_crop_type_3 = (self.compute_harvest_index & (self.var.CropType == 3))
        HI = self.compute_harvest_index_with_logistic_growth(time=np.minimum(self.var.HIt, self.var.tLinSwitch))
        self.var.PctLagPhase[use_logistic_growth_crop_type_3] = (
            100
            * np.divide(
                self.var.HIt,
                self.var.tLinSwitch,
                out=np.zeros_like(self.var.HIt),
                where=self.var.tLinSwitch!=0)
        )[use_logistic_growth_crop_type_3]
        self.var.PctLagPhase[use_linear_growth] = 100        
        self.var.HI[compute_harvest_index_crop_type_3] = HI[compute_harvest_index_crop_type_3]
        self.var.HI[use_linear_growth] += (self.var.dHILinear * (self.var.HIt - self.var.tLinSwitch))[use_linear_growth]
        
    def limit_harvest_index(self):
        """Function to limit HI.

        If HI is within 0.004 of HI0, or exceeds HI0 by any amount, 
        set to HI0. If HI is within 0.004 of HIini, or less than 
        HIini by any amount, set to zero.
        """
        apply_upper_limit = (self.var.HI + 0.004) >= self.var.HI0
        apply_lower_limit = self.var.HI <= (self.var.HIini + 0.004)
        self.var.HI[apply_upper_limit] = self.var.HI0[apply_upper_limit]
        self.var.HI[apply_lower_limit] = 0.
        
    def dynamic(self):        
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()            
        self.is_yield_formation_period()
        self.compute_harvest_index_time()
        
        self.compute_harvest_index = (
            self.var.GrowingSeasonIndex
            & (self.var.HIt > 0)
            & (self.var.CCprev > (self.var.CCmin * self.var.CCx)))
        
        if np.any(self.compute_harvest_index):
            self.compute_harvest_index_crop_type_1_or_2()
            self.compute_harvest_index_crop_type_3()
            self.limit_harvest_index()
        
        self.var.HI[np.logical_not(self.compute_harvest_index)] = 0
        self.var.PctLagPhase[np.logical_not(self.compute_harvest_index)] = 0

class HarvestIndexAdjusted(object):
    """Class to represent adjusted harvest index.

    The harvest index for the current day is adjusted for stress
    effects, as detailed in AquaCrop manual Ch3, section 3.12.3
    """
    def __init__(self, HarvestIndex_variable):
        self.var = HarvestIndex_variable

    def initial(self):
        arr_ones = np.ones((self.var.nFarm, self.var.nCrop, self.var.nCell))
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.Fpre = np.copy(arr_ones)
        self.var.Fpost = np.copy(arr_ones)
        self.var.fpost_dwn = np.copy(arr_ones)
        self.var.fpost_upp = np.copy(arr_ones)
        self.var.Fpol = np.copy(arr_zeros)
        self.var.sCor1 = np.copy(arr_zeros)
        self.var.sCor2 = np.copy(arr_zeros)
        # self.var.HI = np.copy(arr_zeros)
        self.var.HIadj = np.copy(arr_zeros)
        self.var.PreAdj = np.copy(arr_zeros.astype(bool))  # not currently used
        
    def reset_initial_conditions(self):
        self.var.Fpre[self.var.GrowingSeasonDayOne] = 1
        self.var.Fpost[self.var.GrowingSeasonDayOne] = 1
        self.var.fpost_dwn[self.var.GrowingSeasonDayOne] = 1
        self.var.fpost_upp[self.var.GrowingSeasonDayOne] = 1
        self.var.Fpol[self.var.GrowingSeasonDayOne] = 0
        self.var.sCor1[self.var.GrowingSeasonDayOne] = 0
        self.var.sCor2[self.var.GrowingSeasonDayOne] = 0
        # self.var.HI[self.var.GrowingSeasonDayOne] = 0
        self.var.HIadj[self.var.GrowingSeasonDayOne] = 0
        self.var.PreAdj[self.var.GrowingSeasonDayOne] = False  # not currently used

    def compute_HI_adjustment_factor_for_failure_of_pollination(self):
        """Function to calculate adjustment to harvest index for 
        failure of pollination due to water or temperature stress
        """
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        FracFlow = np.copy(arr_zeros)
        t1 = np.copy(arr_zeros)
        t2 = np.copy(arr_zeros)
        F1 = np.copy(arr_zeros)
        F2 = np.copy(arr_zeros)
        F = np.copy(arr_zeros)

        cond0 = (self.var.GrowingSeasonIndex & self.var.YieldForm & (self.var.CropType == 3) & (self.var.HIt > 0) & (self.var.HIt <= self.var.FloweringCD))

        # Fractional flowering on previous day
        # cond1 = (HIt > 0)
        t1[cond0] = self.var.HIt[cond0] - 1
        cond11 = (cond0 & (t1 > 0))
        t1pct = 100 * np.divide(t1, self.var.FloweringCD, out=np.copy(arr_zeros), where=self.var.FloweringCD!=0)
        t1pct = np.clip(t1pct, 0, 100)

        # AquaCrop manual Ch3, Eqn 3.12i
        F1[cond11] = (0.00558 * np.exp(0.63 * np.log(t1pct, out=np.copy(arr_zeros), where=t1pct>0)) - (0.000969 * t1pct) - 0.00383)[cond11]
        F1 = np.clip(F1, 0, None)

        # Fractional flowering on current day
        t2[cond0] = self.var.HIt[cond0]
        cond12 = (cond0 & (t2 > 0))
        t2pct = 100 * np.divide(t2, self.var.FloweringCD, out=np.copy(arr_zeros), where=self.var.FloweringCD!=0)
        t2pct = np.clip(t2pct, 0, 100)
        F2[cond12] = (0.00558 * np.exp(0.63 * np.log(t2pct, out=np.copy(arr_zeros), where=t2pct>0)) - (0.000969 * t2pct) - 0.00383)[cond12]
        F2 = np.clip(F2, 0, None)

        # Weight values
        cond13 = (cond0 & (np.abs(F1 - F2) >= 0.0000001))
        F[cond13] = (100 * np.divide(((F1 + F2) / 2), self.var.FloweringCD, out=np.copy(arr_zeros), where=self.var.FloweringCD!=0))[cond13]
        FracFlow[cond13] = F[cond13]

        # Calculate pollination adjustment for current day
        dFpol = np.copy(arr_zeros)
        cond2 = (cond0 & (self.var.CC >= self.var.CCmin))
        Ks = np.minimum(self.var.Ksw_Pol, self.var.Kst_PolC, self.var.Kst_PolH)
        dFpol[cond2] = (Ks * FracFlow * (1 + (self.var.exc / 100)))[cond2]

        # Calculate pollination adjustment to dateppp
        self.var.Fpol += dFpol
        self.var.Fpol = np.clip(self.var.Fpol, None, 1)

    def compute_HI_adjustment_factor_for_pre_anthesis_water_stress(self):
        """Function to calculate adjustment to harvest index for 
        pre-anthesis water stress
        """
        cond0 = (self.var.GrowingSeasonIndex & self.var.YieldForm & (self.var.HIt >= 0) & ((self.var.CropType == 2) | (self.var.CropType == 3)) & np.logical_not(self.var.PreAdj))
        self.var.PreAdj[cond0] = True

        # Calculate adjustment
        Br = np.divide(self.var.B, self.var.B_NS, out=np.zeros_like(self.var.B_NS), where=self.var.B_NS!=0)
        Br_range = np.log(self.var.dHI_pre, out=np.zeros_like(self.var.dHI_pre), where=self.var.dHI_pre>0) / 5.62
        Br_upp = 1
        Br_low = 1 - Br_range
        Br_top = Br_upp - (Br_range / 3)

        # Get biomass ratio
        ratio_low_divd = (Br - Br_low)
        ratio_low_divs = (Br_top - Br_low)
        ratio_low = np.divide(ratio_low_divd, ratio_low_divs, out=np.zeros_like(ratio_low_divs), where=ratio_low_divs!=0)
        ratio_upp_divd = (Br - Br_top)
        ratio_upp_divs = (Br_upp - Br_top)
        ratio_upp = np.divide(ratio_upp_divd, ratio_upp_divs, out=np.zeros_like(ratio_upp_divs), where=ratio_upp_divs!=0)

        # Calculate adjustment factor
        cond1 = (cond0 & ((Br >= Br_low) & (Br < Br_top)))
        self.var.Fpre[cond1] = (1 + (((1 + np.sin((1.5 - ratio_low) * np.pi)) / 2) * (self.var.dHI_pre / 100)))[cond1]
        cond2 = (cond0 & np.logical_not(cond1) & ((Br > Br_top) & (Br <= Br_upp)))
        self.var.Fpre[cond2] = (1 + (((1 + np.sin((0.5 + ratio_upp) * np.pi)) / 2) * (self.var.dHI_pre / 100)))[cond2]
        cond3 = (cond0 & np.logical_not(cond1 | cond2))
        self.var.Fpre[cond3] = 1

        # No green canopy left at start of flowering so no harvestable crop
        # will develop
        cond3 = (cond0 & (self.var.CC <= 0.01))
        self.var.Fpre[cond3] = 0

    def compute_HI_adjustment_factor_for_leaf_expansion(self):
        pass

    def compute_HI_adjustment_factor_for_stomatal_closure(self):
        pass
    
    def compute_HI_adjustment_factor_for_post_anthesis_water_stress(self):
        """Function to calculate adjustment to harvest index for 
        post-anthesis water stress
        """
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        cond0 = (
            self.var.GrowingSeasonIndex
            & self.var.YieldForm
            & (self.var.HIt > 0)
            & ((self.var.CropType == 2) | (self.var.CropType == 3))
        )

        # 1 Adjustment for leaf expansion
        tmax1 = self.var.CanopyDevEndCD - self.var.HIstartCD
        DAP = self.var.DAP - self.var.DelayedCDs
        cond1 = (cond0 & (DAP <= (self.var.CanopyDevEndCD + 1)) & (tmax1 > 0) & (self.var.Fpre > 0.99) & (self.var.CC > 0.001) & (self.var.a_HI > 0))
        dCor = (1 + np.divide((1 - self.var.Ksw_Exp), self.var.a_HI, out=np.copy(arr_zeros), where=self.var.a_HI!=0))
        self.var.sCor1[cond1] += np.divide(dCor, tmax1, out=np.copy(arr_zeros), where=tmax1!=0)[cond1]
        DayCor = (DAP - 1 - self.var.HIstartCD)
        self.var.fpost_upp[cond1] = (np.divide(tmax1, DayCor, out=np.copy(arr_zeros), where=DayCor!=0) * self.var.sCor1)[cond1]

        # 2 Adjustment for stomatal closure
        tmax2 = np.copy(self.var.YldFormCD)
        cond2 = (cond0 & (DAP <= (self.var.HIendCD + 1)) & (tmax2 > 0) & (self.var.Fpre > 0.99) & (self.var.CC > 0.001) & (self.var.b_HI > 0))
        dCor = ((np.exp(0.1 * np.log(self.var.Ksw_Sto, out=np.zeros_like(self.var.Ksw_Sto), where=self.var.Ksw_Sto!=0))) * (1 - np.divide((1 - self.var.Ksw_Sto), self.var.b_HI, out=np.copy(arr_zeros), where=self.var.b_HI!=0)))
        self.var.sCor2[cond2] += np.divide(dCor, tmax2, out=np.copy(arr_zeros), where=tmax2!=0)[cond2]
        DayCor = (DAP - 1 - self.var.HIstartCD)
        self.var.fpost_dwn[cond2] = (np.divide(tmax2, DayCor, out=np.copy(arr_zeros), where=DayCor!=0) * self.var.sCor2)[cond2]

        # Determine total multiplier
        cond3 = (cond0 & (tmax1 == 0) & (tmax2 == 0))
        self.var.Fpost[cond3] = 1
        cond4 = (cond0 & np.logical_not(cond3))
        cond41 = (cond4 & (tmax2 == 0))
        self.var.Fpost[cond41] = self.var.fpost_upp[cond41]
        cond42 = (cond4 & (tmax1 <= tmax2) & np.logical_not(cond41))
        self.var.Fpost[cond42] = (self.var.fpost_dwn * np.divide(((tmax1 * self.var.fpost_upp) + (tmax2 - tmax1)), tmax2, out=np.copy(arr_zeros), where=tmax2!=0))[cond42]
        cond43 = (cond4 & np.logical_not(cond41 | cond42))
        self.var.Fpost[cond43] = (self.var.fpost_upp * np.divide(((tmax2 * self.var.fpost_dwn) + (tmax1 - tmax2)), tmax2, out=np.copy(arr_zeros), where=tmax2!=0))[cond43]
    
    def dynamic(self):        
        """Function to simulate build up of harvest index"""

        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()

        adjust_harvest_index = (
            self.var.GrowingSeasonIndex
            & self.var.YieldForm
            & (self.var.HIt >= 0)
        )

        adjust_harvest_index_crop_type_1 = (
            adjust_harvest_index
            & (self.var.CropType == 1)
        )
               
        adjust_harvest_index_crop_type_2_or_3 = (
            adjust_harvest_index
            & ((self.var.CropType == 2)
               | (self.var.CropType == 3))
        )

        adjust_harvest_index_crop_type_2 = (
            adjust_harvest_index_crop_type_2_or_3
            & (self.var.CropType == 2)
        )
        
        adjust_harvest_index_crop_type_3 = (
            adjust_harvest_index_crop_type_2_or_3
            & (self.var.CropType == 3)
        )
        
        # Determine adjustment for water stress before anthesis
        self.compute_HI_adjustment_factor_for_pre_anthesis_water_stress()
        self.compute_HI_adjustment_factor_for_failure_of_pollination()
        self.compute_HI_adjustment_factor_for_post_anthesis_water_stress()
        
        HImax = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        HImax[adjust_harvest_index_crop_type_3] = (self.var.Fpol * self.var.HI0)[adjust_harvest_index_crop_type_3]
        HImax[adjust_harvest_index_crop_type_2] = self.var.HI0[adjust_harvest_index_crop_type_2]
                               
        # Limit HI to maximum allowable increase due to pre- and post-anthesis
        # water stress combinations
        HImult = (self.var.Fpre * self.var.Fpost)
        HImult = np.clip(HImult, 1., (1 + (self.var.dHI0 / 100)))
        HI = np.minimum(HImax, self.var.HI)
        self.var.HIadj[adjust_harvest_index_crop_type_2_or_3] = (HImult * HI)[adjust_harvest_index_crop_type_2_or_3]

        # Leafy vegetable crops - no adjustment, harvest index equal to
        # reference value for current day
        self.var.HIadj[adjust_harvest_index_crop_type_1] = self.var.HI[adjust_harvest_index_crop_type_1]
        self.var.HIadj[np.logical_not(self.var.GrowingSeasonIndex)] = 0

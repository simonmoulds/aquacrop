#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
logger = logging.getLogger(__name__)

import aquacrop_fc

class BiomassAccumulation(object):
    def __init__(self, BiomassAccumulation_variable):
        self.var = BiomassAccumulation_variable

    def initial(self):
        arr_zeros = np.zeros((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))
        self.var.B = np.copy(arr_zeros)
        self.var.B_NS = np.copy(arr_zeros)

    def dynamic(self):
        self.dynamic_numpy()
        
    def dynamic_fortran(self):
        aquacrop_fc.biomass_accumulation_w.update_biomass_accum_w(
            self.var.model.etref.values.T,
            self.var.TrAct.T,
            self.var.TrPot_NS.T,
            self.var.B.T,
            self.var.B_NS.T,
            self.var.BioTempStress.T,
            self.var.GDD.T,
            self.var.GDD_up.T,
            self.var.GDD_lo.T,
            self.var.PolHeatStress.T,
            self.var.model.tmax.values.T,
            self.var.Tmax_up.T,
            self.var.Tmax_lo.T,
            self.var.fshape_b.T,
            self.var.PolColdStress.T,
            self.var.model.tmin.values.T,
            self.var.Tmin_up.T,
            self.var.Tmin_lo.T,
            self.var.HI.T,
            self.var.PctLagPhase.T,
            self.var.YldFormCD.T,
            self.var.WP.T,
            self.var.WPy.T,
            self.var.fCO2.T,
            self.var.HIstartCD.T,
            self.var.DelayedCDs.T,
            self.var.DAP.T,
            self.var.CropType.T,
            self.var.Determinant.T,
            self.var.GrowingSeasonIndex.T,
            self.var.nFarm,
            self.var.nCrop,
            self.var.domain.nxy
        )
    
    def dynamic_numpy(self):
        """Function to calculate biomass accumulation"""
        if np.any(self.var.GrowingSeasonDayOne):
            self.reset_initial_conditions()
        WPadj = self.var.WP.copy()
        self.adjust_wp_for_types_of_product_synthesized(WPadj)
        self.adjust_wp_for_co2_effects(WPadj)
        self.adjust_wp_for_soil_fertility(WPadj)
        dB_NS = self.compute_biomass_accumulation_on_current_day(WPadj, self.var.TrPot_NS)
        dB = self.compute_biomass_accumulation_on_current_day(WPadj, self.var.TrAct)
        self.var.B +=dB
        self.var.B_NS += dB_NS
        self.var.B[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        self.var.B_NS[np.logical_not(self.var.GrowingSeasonIndex)] = 0
        
    def reset_initial_conditions(self):
        self.var.B[self.var.GrowingSeasonDayOne] = 0
        self.var.B_NS[self.var.GrowingSeasonDayOne] = 0

    def compute_fswitch(self, require_adjustment):
        """Reduction coefficient for types of products synthesized.
        """
        fswitch = np.ones((self.var.nFarm, self.var.nCrop, self.var.domain.nxy))        
        determinant_crop = (require_adjustment & (self.var.Determinant == 1))
        fswitch[determinant_crop] = (self.var.PctLagPhase / 100)[determinant_crop]
        indeterminant_crop = (require_adjustment & np.logical_not(determinant_crop))
        partial_adjustment = (indeterminant_crop & (self.var.HIt < (self.var.YldFormCD / 3)))
        fswitch[partial_adjustment] = np.divide(
            self.var.HIt.astype(np.float64),
            (self.var.YldFormCD.astype(np.float64) / 3.),
            out=np.zeros_like(self.var.YldFormCD.astype(np.float64)),
            where=self.var.YldFormCD != 0
        )[partial_adjustment]
        return fswitch
        
    def adjust_wp_for_types_of_product_synthesized(self, WPadj):
        """Adjust WP for types of product synthesized.

        From AquaCrop documentation, Ch3:

        "If products that are rich in in lipids or proteins are synthesized
        during yield formation, more energy per unit dry weight is required
        for the synthesis of carbohydrates (Azam-Ali and Squire, 2002). As
        a consequence, the biomass water productivity during yield formation
        needs to be adjusted for the types of products synthesized during
        yield formation:

            WP*adj = fyield.WP* (Eq 3.11g)"

        The types of crops affected by this process are types 2 and 3 
        (2-Root/tuber; 3-Fruit/grain)
        """
        require_adjustment = (
            self.var.GrowingSeasonIndex
            & (((self.var.CropType == 2) | (self.var.CropType == 3))
               & (self.var.HI > 0))
        )
        fswitch = self.compute_fswitch(require_adjustment)        
        WPadj[require_adjustment] = (
            self.var.WP
            * (1 - (1 - self.var.WPy / 100)
               * fswitch)
        )[require_adjustment]
    
    def adjust_wp_for_co2_effects(self, WPadj):
        WPadj *= self.var.fCO2

    def adjust_wp_for_soil_fertility(self, WPadj):
        # TODO: see AquaCrop manual, Ch3, Eq 3.11h
        WPadj *= 1.

    def compute_biomass_accumulation_on_current_day(self, WPadj, Tr):
        return WPadj * (Tr / self.var.model.etref.values) * self.var.Kst_Bio
        

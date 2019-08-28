#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.BiomassAccumulation import BiomassAccumulation

@pytest.fixture(scope='session')
def biomass_accumulation():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    biomass_accumulation = BiomassAccumulation(mock_aquacrop_model)
    biomass_accumulation.initial()
    return biomass_accumulation

def test_compute_fswitch_1(biomass_accumulation):
    # example based on Ch7_Ex1a_Tunis_Wheat/1979
    # time step 94
    biomass_accumulation.var.YldFormCD = np.full((1,1,1), 90)
    biomass_accumulation.var.Determinant = np.full((1,1,1), 1)
    biomass_accumulation.var.HIt = np.full((1,1,1), 4.)
    biomass_accumulation.var.HI = np.full((1,1,1), 0.01404057)
    biomass_accumulation.var.PctLagPhase = np.full((1,1,1), 14.81481481)
    fswitch = biomass_accumulation.compute_fswitch(
        require_adjustment = np.full((1,1,1), True)
    )
    np.testing.assert_array_almost_equal(fswitch, 0.148148)

def test_compute_fswitch_2(biomass_accumulation):
    # example based on Ch9_Ex1_Brussels_Potato/2000
    # time step 44
    biomass_accumulation.var.GrowingSeasonIndex = np.full((1,1,1), True)
    biomass_accumulation.var.CropType = np.full((1,1,1), 2)
    biomass_accumulation.var.WP = np.full((1,1,1), 18)
    biomass_accumulation.var.WPy = np.full((1,1,1), 100)
    biomass_accumulation.var.YldFormCD = np.full((1,1,1), 75)
    biomass_accumulation.var.Determinant = np.full((1,1,1), 0)
    biomass_accumulation.var.HIt = np.full((1,1,1), 4.)
    biomass_accumulation.var.HI = np.full((1,1,1), 0.01548746)
    biomass_accumulation.var.PctLagPhase = np.full((1,1,1), 100)
    fswitch = biomass_accumulation.compute_fswitch(
        require_adjustment = np.full((1,1,1), True)
    )
    np.testing.assert_array_almost_equal(fswitch, 0.16)

def test_adjust_wp_for_types_of_product_synthesized_1(biomass_accumulation):
    WPadj = np.full((1,1,1), 18.)
    biomass_accumulation.adjust_wp_for_types_of_product_synthesized(WPadj)
    np.testing.assert_array_almost_equal(WPadj, 18.)

def test_adjust_wp_for_types_of_product_synthesized_2(biomass_accumulation):
    biomass_accumulation.var.GrowingSeasonIndex = np.full((1,1,1), True)
    biomass_accumulation.var.CropType = np.full((1,1,1), 3)
    biomass_accumulation.var.WP = np.full((1,1,1), 15)
    biomass_accumulation.var.WPy = np.full((1,1,1), 100)
    biomass_accumulation.var.YldFormCD = np.full((1,1,1), 90)
    biomass_accumulation.var.Determinant = np.full((1,1,1), 1)
    biomass_accumulation.var.HIt = np.full((1,1,1), 4.)
    biomass_accumulation.var.HI = np.full((1,1,1), 0.01404057)
    biomass_accumulation.var.PctLagPhase = np.full((1,1,1), 14.81481481)
    WPadj = np.full((1,1,1), 15.)
    biomass_accumulation.adjust_wp_for_types_of_product_synthesized(WPadj)
    np.testing.assert_array_almost_equal(WPadj, 15.)

def test_adjust_wp_for_co2_effects(biomass_accumulation):
    # hypothetical
    WPadj = np.full((1,1,1), 15.)
    biomass_accumulation.var.fCO2 = np.full((1,1,1), 0.8)
    biomass_accumulation.adjust_wp_for_co2_effects(WPadj)
    np.testing.assert_array_almost_equal(WPadj, 12.)

def test_adjust_wp_for_soil_fertility(biomass_accumulation):
    pass

def test_compute_biomass_accumulation_on_current_day(biomass_accumulation):
    biomass_accumulation.var.weather.referencePotET = np.full((1,1,1), 1.2)
    biomass_accumulation.var.Kst_Bio = np.full((1,1,1), 0.87)
    WPadj = np.full((1,1,1), 13.73690587)
    dB = biomass_accumulation.compute_biomass_accumulation_on_current_day(
        WPadj=WPadj,
        Tr=np.full((1,1,1), 1.28304265)
    )
    np.testing.assert_array_almost_equal(dB, 12.778151)

def test_dynamic(biomass_accumulation):
    pass
    

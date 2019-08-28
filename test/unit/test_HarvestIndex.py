import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.HarvestIndex import HarvestIndex

@pytest.fixture(scope='session')
def harvest_index():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    mock_aquacrop_model.HIini = np.full((1,1,1), 0.01)
    mock_aquacrop_model.HI0 = np.full((1,1,1), 0.48)
    mock_aquacrop_model.HIGC = np.full((1,1,1), 0.087)
    mock_aquacrop_model.HIstart = np.full((1,1,1), 1250)
    mock_aquacrop_model.HIstartCD = np.full((1,1,1), 89.)
    mock_aquacrop_model.GrowingSeasonIndex = np.full((1,1,1), True)    
    harvest_index = HarvestIndex(mock_aquacrop_model)
    harvest_index.initial()
    return harvest_index

def test_compute_harvest_index_with_logistic_growth(harvest_index):
    # test from Ch7_Ex1a_Tunis_Wheat example, time step 100
    time = np.full((1,1,1), 10.)
    HI = harvest_index.compute_harvest_index_with_logistic_growth(time)
    np.testing.assert_array_almost_equal(HI, 0.0231988)

def test_is_yield_formation_period_1(harvest_index):
    # time step 88
    harvest_index.var.time_since_germination = np.full((1,1,1), 1247.)
    harvest_index.is_yield_formation_period()
    np.testing.assert_array_equal(harvest_index.var.YieldForm, False)    

def test_is_yield_formation_period_2(harvest_index):
    # time step 89
    harvest_index.var.time_since_germination = np.full((1,1,1), 1255.5)
    harvest_index.is_yield_formation_period()
    np.testing.assert_array_equal(harvest_index.var.YieldForm, True)
    
def test_compute_harvest_index_time(harvest_index):
    # time step 100
    harvest_index.var.DAP = np.full((1,1,1), 100.)
    harvest_index.var.DelayedCDs = np.full((1,1,1), 0.)
    harvest_index.compute_harvest_index_time()
    np.testing.assert_array_almost_equal(harvest_index.var.HIt, 10.)
    
def test_compute_harvest_index_crop_type_1_or_2(harvest_index):
    pass

def test_compute_harvest_index_crop_type_3_1(harvest_index):
    # time step 100
    harvest_index.compute_harvest_index = np.full((1,1,1), True)
    harvest_index.var.CropType = np.full((1,1,1), 3)
    harvest_index.var.HIt = np.full((1,1,1), 10.)
    harvest_index.var.tLinSwitch = np.full((1,1,1), 27)
    harvest_index.var.dHILinear = np.full((1,1,1), 0.00623044)
    harvest_index.compute_harvest_index_crop_type_3()
    np.testing.assert_array_almost_equal(harvest_index.var.HI, 0.0231988)

def test_compute_harvest_index_crop_type_3_2(harvest_index):
    np.testing.assert_array_almost_equal(harvest_index.var.PctLagPhase, 37.03703704)

def test_compute_harvest_index_crop_type_3_3(harvest_index):
    # time step 120
    harvest_index.var.HIt = np.full((1,1,1), 30.)
    harvest_index.compute_harvest_index_crop_type_3()
    np.testing.assert_array_almost_equal(harvest_index.var.HI, 0.10617345)

def test_compute_harvest_index_crop_type_3_4(harvest_index):
    np.testing.assert_array_almost_equal(harvest_index.var.PctLagPhase, 100.)
    
def test_limit_harvest_index_1(harvest_index):
    # hypothetical
    harvest_index.var.HI = np.full((1,1,1), 0.5)
    harvest_index.limit_harvest_index()
    np.testing.assert_array_almost_equal(harvest_index.var.HI, 0.48)

def test_limit_harvest_index_2(harvest_index):
    # hypothetical
    harvest_index.var.HI = np.full((1,1,1), 0.476)
    harvest_index.limit_harvest_index()
    np.testing.assert_array_almost_equal(harvest_index.var.HI, 0.48)

def test_limit_harvest_index_3(harvest_index):
    # hypothetical
    harvest_index.var.HI = np.full((1,1,1), 0.475)
    harvest_index.limit_harvest_index()
    np.testing.assert_array_almost_equal(harvest_index.var.HI, 0.475)
    
def test_dynamic(harvest_index):
    pass

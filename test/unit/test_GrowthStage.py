import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.GrowthStage import GrowthStage

@pytest.fixture(scope='session')
def growth_stage():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    mock_aquacrop_model.GrowingSeasonIndex = np.full((1,1,1), True)    
    mock_aquacrop_model.CalendarType = 2
    mock_aquacrop_model.Canopy10Pct = np.full((1,1,1), 229.)
    mock_aquacrop_model.MaxCanopy = np.full((1,1,1), 1186.)
    mock_aquacrop_model.Senescence = np.full((1,1,1), 1700.)
    growth_stage = GrowthStage(mock_aquacrop_model)
    growth_stage.initial()
    return growth_stage

def test_dynamic_1(growth_stage):
    # example from Ch7_Ex1a_Tunis_Wheat example, time step 10
    growth_stage.var.GDDcum = np.full((1,1,1), 217.)
    growth_stage.var.DelayedGDDs = np.full((1,1,1), 0.)
    growth_stage.dynamic()
    np.testing.assert_array_almost_equal(growth_stage.var.GrowthStage, 1)

def test_dynamic_2(growth_stage):
    # time step 11
    growth_stage.var.GDDcum = np.full((1,1,1), 236.5)
    growth_stage.var.DelayedGDDs = np.full((1,1,1), 0.)
    growth_stage.dynamic()
    np.testing.assert_array_almost_equal(growth_stage.var.GrowthStage, 2)

def test_dynamic_3(growth_stage):
    # time step 82
    growth_stage.var.GDDcum = np.full((1,1,1), 1196.)
    growth_stage.var.DelayedGDDs = np.full((1,1,1), 0.)
    growth_stage.dynamic()
    np.testing.assert_array_almost_equal(growth_stage.var.GrowthStage, 3)

def test_dynamic_4(growth_stage):
    # time step 126
    growth_stage.var.GDDcum = np.full((1,1,1), 1704.)
    growth_stage.var.DelayedGDDs = np.full((1,1,1), 0.)
    growth_stage.dynamic()
    np.testing.assert_array_almost_equal(growth_stage.var.GrowthStage, 4)    

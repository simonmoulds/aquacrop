import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.CropParameters import CropParameters

@pytest.fixture(scope='session')
def crop_parameters():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    # crop_parameters = CropParameters(mock_aquacrop_model)
    # crop_parameters.initial()
    # return crop_parameters

def test_adjust_planting_and_harvesting_date(crop_parameters):
    pass

def test_update_growing_season(crop_parameters):
    pass

def test_compute_crop_parameters(crop_parameters):
    pass

def test_compute_water_productivity_adjustment_factor(crop_parameters):
    pass

def test_calculate_HI_linear(crop_parameters):
    pass

def test_calculate_HIGC(crop_parameters):
    pass

def test_compute_crop_calendar(crop_parameters):
    pass

def test_update_parameters(crop_parameters):
    pass

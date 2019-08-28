import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.Irrigation import Irrigation

@pytest.fixture(scope='session')
def irrigation():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    irrigation = Irrigation(mock_aquacrop_model)
    irrigation.initial()
    return irrigation

def test_compute_irrigation_depth_soil_moisture_threshold(irrigation):
    pass

def test_compute_irrigation_depth_fixed_interval(irrigation):
    pass

def test_dynamic(irrigation):
    pass

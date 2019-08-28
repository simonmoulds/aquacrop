import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.RainfallPartition import RainfallPartition

@pytest.fixture(scope='session')
def rainfall_partition():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    rainfall_partition = RainfallPartition(mock_aquacrop_model)
    rainfall_partition.initial()
    return rainfall_partition

def test_compute_relative_wetness_of_soil(rainfall_partition):
    pass

def test_adjust_curve_number(rainfall_partition):
    pass

def test_compute_potential_maximum_soil_water_retention(rainfall_partition):
    pass

def test_dynamic(rainfall_partition):
    pass

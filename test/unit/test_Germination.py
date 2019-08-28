import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.Germination import Germination

# TODO: first need to test RootZoneWater functions

@pytest.fixture(scope='session')
def germination():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    mock_aquacrop_model.GrowingSeasonIndex = np.full((1,1,1), True)
    mock_aquacrop_model.CalendarType = 2
    
    germination = Germination(mock_aquacrop_model)
    germination.initial()
    return germination

def test_compute_proportional_water_content_in_root_zone(germination):
    pass

def test_increment_delayed_growth_time_counters(germination):
    pass

def update_ageing_days_counter(germination):
    pass

def test_dynamic(germination):
    pass

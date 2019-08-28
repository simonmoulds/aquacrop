import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.PreIrrigation import PreIrrigation

@pytest.fixture(scope='session')
def pre_irrigation():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    pre_irrigation = PreIrrigation(mock_aquacrop_model)
    pre_irrigation.initial()
    return pre_irrigation
    
def test_dynamic(pre_irrigation):
    pass

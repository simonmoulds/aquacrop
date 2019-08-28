import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.Infiltration import Infiltration

@pytest.fixture(scope='session')
def infiltration():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    mock_aquacrop_model.Bunds = np.full((1,1,1), 0)
    mock_aquacrop_model.zBund = np.full((1,1,1), 0.)
    mock_aquacrop_model.BundWater = np.full((1,1,1), 0.)
    infiltration = Infiltration(mock_aquacrop_model)
    infiltration.initial()
    return infiltration
    
def test_dynamic(infiltration):
    pass

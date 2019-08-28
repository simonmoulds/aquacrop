import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.Drainage import Drainage

@pytest.fixture(scope='session')
def drainage():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    mock_aquacrop_model.nComp = 12
    drainage = Drainage(mock_aquacrop_model)
    drainage.initial()
    return drainage

def test_dynamic(drainage):
    pass

import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.CropYield import CropYield

@pytest.fixture(scope='session')
def crop_yield():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    crop_yield = CropYield(mock_aquacrop_model)
    crop_yield.initial()
    return crop_yield

def test_dynamic(crop_yield):
    pass

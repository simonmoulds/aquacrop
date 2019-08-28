import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.Evapotranspiration import Evapotranspiration

@pytest.fixture(scope='session')
def evapotranspiration():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    evapotranspiration = Evapotranspiration(mock_aquacrop_model)
    evapotranspiration.initial()
    return evapotranspiration

def test_dynamic(evapotranspiration):
    # hypothetical
    evapotranspiration.var.Epot = 5.
    evapotranspiration.var.Tpot = 2.5
    evapotranspiration.dynamic()
    np.testing.assert_array_almost_equal(evapotranspiration.var.ETpot, 7.5)

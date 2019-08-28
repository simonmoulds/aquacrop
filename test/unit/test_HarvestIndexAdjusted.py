import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.HarvestIndex import HarvestIndexAdjusted

@pytest.fixture(scope='session')
def harvest_index_adj():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    mock_aquacrop_model.HIini = np.full((1,1,1), 0.01)
    mock_aquacrop_model.HI0 = np.full((1,1,1), 0.48)
    mock_aquacrop_model.HIGC = np.full((1,1,1), 0.087)
    mock_aquacrop_model.HIstart = np.full((1,1,1), 1250)
    mock_aquacrop_model.HIstartCD = np.full((1,1,1), 89.)
    mock_aquacrop_model.GrowingSeasonIndex = np.full((1,1,1), True)    
    harvest_index_adj = HarvestIndexAdjusted(mock_aquacrop_model)
    harvest_index_adj.initial()
    return harvest_index_adj
    
def test_dynamic(harvest_index_adj):
    pass

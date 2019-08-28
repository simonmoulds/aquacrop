import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.GrowingDegreeDay import GrowingDegreeDay

@pytest.fixture(scope='session')
def growing_degree_day():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    mock_aquacrop_model.weather.tmin = np.full((1,1,1), 17.)
    mock_aquacrop_model.weather.tmax = np.full((1,1,1), 21.)
    mock_aquacrop_model.Tupp = np.full((1,1,1), 26.)
    mock_aquacrop_model.Tbase = np.full((1,1,1), 0.)
    mock_aquacrop_model.GrowingSeasonIndex = np.full((1,1,1), True)
    mock_aquacrop_model.GDDmethod = 3
    growing_degree_day = GrowingDegreeDay(mock_aquacrop_model)
    growing_degree_day.initial()
    return growing_degree_day

def test_growing_degree_day(growing_degree_day):
    growing_degree_day.growing_degree_day()
    np.testing.assert_array_almost_equal(
        growing_degree_day.var.GDD,
        19.
    )

def test_dynamic(growing_degree_day):
    # example from Ch7_Ex1a_Tunis_Wheat example, time step 10
    growing_degree_day.var.GDDcum = np.full((1,1,1), 198.)
    growing_degree_day.dynamic()
    np.testing.assert_array_almost_equal(
        growing_degree_day.var.GDDcum,
        217.
    )

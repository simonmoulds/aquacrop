import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.CheckGroundwaterTable import CheckGroundwaterTable

@pytest.fixture(scope='session')
def check_gw_table():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    mock_aquacrop_model.nComp = 12    
    mock_aquacrop_model.groundwater.WaterTable = True
    mock_aquacrop_model.th = (
        np.array(
            [0.3,0.3,0.3,0.2875,0.2625,0.2375,0.20625,0.16875,0.15,0.15,0.15,0.15]
        )[None,None,:,None]
    )
    mock_aquacrop_model.th_fc_comp = (
        np.array(
            [0.4,0.4,0.4,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33]
        )[None,None,:,None]
    )
    mock_aquacrop_model.th_fc_adj = np.zeros((1,1,12,1))
    mock_aquacrop_model.th_sat_comp = (
        np.array(
            [0.5,0.5,0.5,0.46,0.46,0.46,0.46,0.46,0.46,0.46,0.46,0.46]
        )[None,None,:,None]
    )
    mock_aquacrop_model.dz = (
        np.array(
            [0.1,0.1,0.1,0.1,0.1,0.1,0.15,0.15,0.15,0.15,0.15,0.2]
        )
    )
    mock_aquacrop_model.groundwater.zGW = np.array([2.])
    check_gw_table = CheckGroundwaterTable(mock_aquacrop_model)
    check_gw_table.initial()
    return check_gw_table

def test_compute_mid_point_of_compartments(check_gw_table):
    zMid = check_gw_table.compute_mid_point_of_compartments()
    np.testing.assert_array_almost_equal(
        zMid[0,0,:,0],
        np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.675,0.825,0.975,1.125,1.275,1.45])
    )

def test_compute_Xmax(check_gw_table):
    Xmax = check_gw_table.compute_Xmax()
    np.testing.assert_array_almost_equal(
        Xmax[0,0,:,0],
        np.array([2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.,2.])
    )

def test_compute_th_fc_adj(check_gw_table):
    # example from Ch7_Ex1a_Tunis_Wheat example, time step 10
    check_gw_table.compute_th_fc_adj()
    np.testing.assert_array_almost_equal(
        check_gw_table.var.th_fc_adj[0,0,:,0],
        np.array([0.400063,0.400563,0.401563,0.333981,0.336581,0.339831,0.344808,0.35212,0.360895,0.371133,0.382833,0.398331])
    )

def test_dynamic(check_gw_table):
    pass

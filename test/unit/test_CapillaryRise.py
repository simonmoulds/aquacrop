#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.CapillaryRise import CapillaryRise

@pytest.fixture(scope='session')
def capillary_rise():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    mock_aquacrop_model.k_sat = np.full((1,1,2,1), 500)
    mock_aquacrop_model.aCR = np.full((1,1,2,1), -0.4536)
    mock_aquacrop_model.bCR = np.full((1,1,2,1), 0.837339749)
    capillary_rise = CapillaryRise(mock_aquacrop_model)
    capillary_rise.initial()
    return capillary_rise

def test_maximum_capillary_rise(capillary_rise):
    capillary_rise.var.zGW = np.full((1,1,1), 5)
    MaxCR = capillary_rise.maximum_capillary_rise(
        k_sat=capillary_rise.var.k_sat[...,-1,:],
        aCR=capillary_rise.var.aCR[...,-1,:],
        bCR=capillary_rise.var.bCR[...,-1,:],
        zGW=capillary_rise.var.zGW,
        z=1.45)
    np.testing.assert_array_almost_equal(MaxCR, 0.387861)

def test_store_water_from_capillary_rise(capillary_rise):
    pass

def test_dynamic(capillary_rise):
    pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from mock import patch, Mock, MagicMock
from aquacrop.CanopyCover import CanopyCover

@pytest.fixture(scope='session')
def canopycover():
    mock_aquacrop_model = Mock()
    mock_aquacrop_model.nFarm = 1
    mock_aquacrop_model.nCrop = 1
    mock_aquacrop_model.nCell = 1
    mock_aquacrop_model.CC0=np.full((1,1,1), 0.0675)
    mock_aquacrop_model.CCx=np.full((1,1,1), 0.96)
    mock_aquacrop_model.CGC=np.full((1,1,1), 0.005)
    mock_aquacrop_model.CDC=np.full((1,1,1), 0.004)
    mock_aquacrop_model.CanopyDevEnd=np.full((1,1,1), 1350)
    mock_aquacrop_model.GrowingSeasonIndex=np.full((1,1,1), True)
    mock_aquacrop_model.Emergence=np.full((1,1,1), 150)
    mock_aquacrop_model.Senescence = np.full((1,1,1), 1700)
    mock_aquacrop_model.Maturity = np.full((1,1,1), 2400)    
    canopycover = CanopyCover(mock_aquacrop_model)
    canopycover.initial()
    return canopycover

def test_canopy_cover_development_1(canopycover):
    CC = canopycover.canopy_cover_development(
        CC0=canopycover.var.CC0,
        CCx=canopycover.var.CCx,
        CGC=canopycover.var.CGC,
        dt=np.full((1,1,1), 18.5))
    np.testing.assert_array_almost_equal(CC, 0.074043)

def test_canopy_cover_development_2(canopycover):
    CC = canopycover.canopy_cover_development(
        CC0=canopycover.var.CC0,
        CCx=canopycover.var.CCx,
        CGC=canopycover.var.CGC,
        dt=np.full((1,1,1), 1204.957))
    np.testing.assert_array_almost_equal(CC, 0.951746)
    
def test_green_canopy_cover_decline(canopycover):
    CC = canopycover.green_canopy_cover_decline(
        CCx=canopycover.var.CCx,
        CDC=canopycover.var.CDC,
        dt=np.full((1,1,1), 8.))
    np.testing.assert_array_almost_equal(CC, 0.958373)

def test_compute_target_cgc(canopycover):
    res = canopycover.compute_target_cgc(
        x=np.full((1,1,1), 1.096933),
        tSum=np.full((1,1,1), 178.5),
        dt=np.full((1,1,1), 20.5))
    np.testing.assert_array_almost_equal(res, 0.000586)

def test_canopy_cover_required_time_cgc_mode(canopycover):
    # time step 9
    canopycover.var.CCprev = np.full((1,1,1), 0.074043)
    tReq = canopycover.canopy_cover_required_time_cgc_mode(
        CC0=canopycover.var.CC0,
        CCx=canopycover.var.CCx,
        CGC=np.full((1,1,1), 0.004481),
        dt=np.full((1,1,1), 20.5),
        tSum=np.full((1,1,1), 178.5))
    np.testing.assert_array_almost_equal(tReq, 20.646822)

def test_canopy_cover_required_time_cdc_mode(canopycover):
    # time step 9
    canopycover.var.CCprev = np.full((1,1,1), 0.074043)
    tReq = canopycover.canopy_cover_required_time_cdc_mode(
        CCx=np.full((1,1,1), 0.),
        CDC=np.full((1,1,1), 0.0001))
    np.testing.assert_array_equal(tReq, 0.)

def test_adjust_CCx(canopycover):
    # time step 9
    canopycover.var.CCprev = np.full((1,1,1), 0.074043)
    CCxAdj = canopycover.adjust_CCx(
        CC0=canopycover.var.CC0,
        CCx=canopycover.var.CCx,
        CGC=np.full((1,1,1), 0.0044812),
        CDC=canopycover.var.CDC,
        dt=np.full((1,1,1), 20.5),
        tSum=np.full((1,1,1), 178.5),
        CanopyDevEnd=canopycover.var.CanopyDevEnd)
    np.testing.assert_array_almost_equal(CCxAdj, 0.9451)

def test_adjust_CGC_for_water_stress(canopycover):
    pass

def test_adjust_CDC_for_water_stress(canopycover):
    # time step 220
    canopycover.var.Ksw_Sen = np.full((1,1,1), 0.985710)
    CDCadj = canopycover.adjust_CDC_for_water_stress()
    np.testing.assert_array_almost_equal(CDCadj, 0.000435)
                
def test_adjust_CDC_late_stage(canopycover):
    pass

def test_is_initial_stage_1(canopycover):
    # hypothetical
    is_initial_stage = canopycover.is_initial_stage(
        CC0=canopycover.var.CC0,
        CCprev=np.full((1,1,1), 0.15),
        tCC=175)
    np.testing.assert_array_equal(is_initial_stage, False)

def test_is_initial_stage_2(canopycover):
    # hypothetical
    is_initial_stage = canopycover.is_initial_stage(
        CC0=canopycover.var.CC0,
        CCprev=np.full((1,1,1), 0.05),
        tCC=175)
    np.testing.assert_array_equal(is_initial_stage, True)

def test_is_crop_development_stage_1(canopycover):
    is_crop_dev_stage = canopycover.is_crop_development_stage(
        CC0=canopycover.var.CC0,
        CCprev=np.full((1,1,1), 0.05),
        tCC=175)
    np.testing.assert_array_equal(is_crop_dev_stage, False)

def test_is_crop_development_stage_2(canopycover):
    is_crop_dev_stage = canopycover.is_crop_development_stage(
        CC0=canopycover.var.CC0,
        CCprev=np.full((1,1,1), 0.15),
        tCC=175)
    np.testing.assert_array_equal(is_crop_dev_stage, True)

def test_is_canopy_approaching_maximum_size_1(canopycover):
    is_approaching_max_size = canopycover.is_canopy_approaching_maximum_size(
        crop_development_stage=np.full((1,1,1), True),
        CCprev=np.full((1,1,1), 0.95),
        CCx=0.96)
    np.testing.assert_array_equal(is_approaching_max_size, True)

def test_is_canopy_approaching_maximum_size_2(canopycover):
    is_approaching_max_size = canopycover.is_canopy_approaching_maximum_size(
        crop_development_stage=np.full((1,1,1), True),
        CCprev=np.full((1,1,1), 0.94),
        CCx=0.96)
    np.testing.assert_array_equal(is_approaching_max_size, False)
    
def test_is_late_stage_1(canopycover):
    res = canopycover.is_late_stage(tCC=1700)
    np.testing.assert_array_equal(res, True)

def test_is_late_stage_2(canopycover):
    res = canopycover.is_late_stage(tCC=1695)
    np.testing.assert_array_equal(res, False)

def test_potential_canopy_development_1(canopycover):
    # initial; time step 8
    canopycover.var.CC_NS = np.full((1,1,1), 0.)
    canopycover.potential_canopy_development(
        dtCC=np.full((1,1,1), 18.5),
        tCC=np.full((1,1,1), 158))
    np.testing.assert_array_almost_equal(canopycover.var.CC_NS, 0.074043)
    
def test_potential_canopy_development_2(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxAct_NS, 0.074043)
    
def test_potential_canopy_development_3(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxW_NS, 0.)
    
def test_potential_canopy_development_4(canopycover):
    # crop development; time step 9
    canopycover.var.CC_NS = np.full((1,1,1), 0.074043)
    canopycover.potential_canopy_development(
        dtCC=np.full((1,1,1), 20.5),
        tCC=np.full((1,1,1), 178.5))
    np.testing.assert_array_almost_equal(canopycover.var.CC_NS, 0.077838)
    
def test_potential_canopy_development_5(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxAct_NS, 0.077838)
    
def test_potential_canopy_development_6(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxW_NS, 0.)
    
def test_potential_canopy_development_7(canopycover):
    # mid stage; time step 98
    canopycover.var.CC_NS = np.full((1,1,1), 0.951248)
    canopycover.potential_canopy_development(
        dtCC=np.full((1,1,1), 10.),
        tCC=np.full((1,1,1), 1353.))
    np.testing.assert_array_almost_equal(canopycover.var.CC_NS, 0.951248)
    
def test_potential_canopy_development_8(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxAct_NS, 0.951248)
    
def test_potential_canopy_development_9(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxW_NS, 0.951248)

def test_potential_canopy_development_10(canopycover):
    # late stage; time step 127
    canopycover.var.CC_NS = np.full((1,1,1), 0.951248)
    canopycover.potential_canopy_development(
        dtCC=np.full((1,1,1), 11.),
        tCC=np.full((1,1,1), 1704.))
    np.testing.assert_array_almost_equal(canopycover.var.CC_NS, 0.)
    
def test_potential_canopy_development_11(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxAct_NS, 0.951248)
    
def test_potential_canopy_development_12(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxW_NS, 0.951248)

def test_actual_canopy_development_1(canopycover):
    # initial; time step 8
    canopycover.var.CCprev = np.full((1,1,1), 0.)
    canopycover.var.CC = np.full((1,1,1), 0.)
    canopycover.var.CC0adj = np.full((1,1,1), 0.0675)
    canopycover.var.Ksw_Exp = np.full((1,1,1), 0.89618495)
    canopycover.actual_canopy_development(
        tCC=np.full((1,1,1), 158.),
        tCCadj=np.full((1,1,1), 158.),
        dtCC=np.full((1,1,1), 18.5))
    np.testing.assert_array_almost_equal(canopycover.var.CC, 0.074043)
    
def test_actual_canopy_development_2(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxAct, 0.074043)
    
def test_actual_canopy_development_3(canopycover):
    # development; time step 9
    canopycover.var.CCprev = np.full((1,1,1), 0.07404301)
    canopycover.var.CC0adj = np.full((1,1,1), 0.0675)
    canopycover.actual_canopy_development(
        tCC=np.full((1,1,1), 178.5),
        tCCadj=np.full((1,1,1), 178.5),
        dtCC=np.full((1,1,1), 20.5))
    np.testing.assert_array_almost_equal(canopycover.var.CC, 0.081168)
    
def test_actual_canopy_development_4(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxAct, 0.081168)

def test_actual_canopy_development_5(canopycover):
    # mid stage; time step 98
    canopycover.var.CCprev = np.full((1,1,1), 0.93596559)
    canopycover.var.CC0adj = np.full((1,1,1), 0.0675)
    canopycover.var.Ksw_Exp = np.full((1,1,1), 0.93678233)
    canopycover.actual_canopy_development(
        tCC=np.full((1,1,1), 1353.),
        tCCadj=np.full((1,1,1), 1353.),
        dtCC=np.full((1,1,1), 10.))
    np.testing.assert_array_almost_equal(canopycover.var.CC, 0.935965)
    
def test_actual_canopy_development_6(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxAct, 0.935965)

def test_actual_canopy_development_7(canopycover):
    # late stage; time step 127
    canopycover.var.CCprev = np.full((1,1,1), 0.935965)
    canopycover.var.CC0adj = np.full((1,1,1), 0.0675)
    canopycover.var.Ksw_Exp = np.full((1,1,1), 0.96982163)
    canopycover.actual_canopy_development(
        tCC=np.full((1,1,1), 1704.),
        tCCadj=np.full((1,1,1), 1704.),
        dtCC=np.full((1,1,1), 11.))
    np.testing.assert_array_almost_equal(canopycover.var.CC, 0.935179)
    
def test_actual_canopy_development_8(canopycover):
    np.testing.assert_array_almost_equal(canopycover.var.CCxAct, 0.935965)

def test_adjust_CCx_for_late_stage_rewatering(canopycover):
    # late stage; time step 127
    canopycover.var.CCprev = np.full((1,1,1), 0.93596559)
    CCxAdj = canopycover.adjust_CCx_for_late_stage_rewatering(
        dt=np.full((1,1,1), -7.))
    np.testing.assert_array_almost_equal(CCxAdj, 0.934622)

def test_adjust_CDC_for_late_stage_rewatering(canopycover):
    CDCadj = canopycover.adjust_CDC_for_late_stage_rewatering(
        dt=np.full((1,1,1), -7.),
        CCxAdj=np.full((1,1,1), 0.93462228))
    np.testing.assert_array_almost_equal(CDCadj, 0.003894)
        
def test_time_step_1(canopycover):
    # hypothetical, calendar day
    canopycover.var.DAP = np.full((1,1,1), 75)
    canopycover.var.GDD = np.full((1,1,1), 11.5)
    canopycover.var.GDDcum = np.full((1,1,1), 1106.)
    canopycover.var.DelayedCDs = np.full((1,1,1), 0.)
    canopycover.var.DelayedGDDs = np.full((1,1,1), 0.)
    canopycover.var.CalendarType = np.full((1,1,1), 1)
    dtCC = canopycover.time_step()
    np.testing.assert_array_equal(dtCC, 1)

def test_time_step_2(canopycover):
    # hypothetical, growing degree day
    canopycover.var.CalendarType = np.full((1,1,1), 2)
    dtCC = canopycover.time_step()
    np.testing.assert_array_almost_equal(dtCC, 11.5)

def test_time_since_planting(canopycover):
    pass

def test_adjust_time_since_planting_1(canopycover):
    # hypothetical, calendar day
    canopycover.var.CalendarType = np.full((1,1,1), 1)
    tCCadj = canopycover.adjust_time_since_planting()
    np.testing.assert_array_equal(tCCadj, 75)
    
def test_adjust_time_since_planting_2(canopycover):
    # hypothetical, growing degree day
    canopycover.var.CalendarType = np.full((1,1,1), 2)
    tCCadj = canopycover.adjust_time_since_planting()
    np.testing.assert_array_almost_equal(tCCadj, 1106.)

def test_is_crop_dead_1(canopycover):
    # hypothetical
    canopycover.var.CC = np.full((1,1,1), 0.001)
    canopycover.var.CropDead = np.full((1,1,1), False)
    crop_dead = canopycover.is_crop_dead(
        tCCadj=np.full((1,1,1), 2400)
    )
    np.testing.assert_array_equal(crop_dead, False)

def test_is_crop_dead_2(canopycover):
    # hypothetical
    canopycover.var.CC = np.full((1,1,1), 0.0009)
    crop_dead = canopycover.is_crop_dead(
        tCCadj=np.full((1,1,1), 2400)
    )
    np.testing.assert_array_equal(crop_dead, True)
    
def test_is_crop_dead_3(canopycover):
    # hypothetical
    crop_dead = canopycover.is_crop_dead(
        tCCadj=np.full((1,1,1), 1350)
    )
    np.testing.assert_array_equal(crop_dead, False)

def test_is_crop_dead_4(canopycover):
    # hypothetical
    crop_dead = canopycover.is_crop_dead(
        tCCadj=np.full((1,1,1), 2405)
    )
    np.testing.assert_array_equal(crop_dead, False)

def test_is_crop_dead_5(canopycover):
    # hypothetical
    canopycover.var.GrowingSeasonIndex = np.full((1,1,1), False)
    crop_dead = canopycover.is_crop_dead(
        tCCadj=np.full((1,1,1), 2400)
    )
    np.testing.assert_array_equal(crop_dead, False)

# TODO: the following tests should be implemented for a case
# where the plant suffers from early senescence
def test_canopy_cover_after_senescence(canopycover):
    pass

def test_adjust_canopy_cover_after_senescence(canopycover):
    pass

def test_update_canopy_cover_after_rewatering_in_late_season_1(canopycover):
    pass

def test_update_canopy_cover_after_rewatering_in_late_season_1(canopycover):
    pass

def test_canopy_senescence_due_to_water_stress(canopycover):
    pass

def test_adjust_canopy_cover_for_microadvective_effects(canopycover):
    pass

def test_dynamic(canopycover):
    pass


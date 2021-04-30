#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

import os
import sys
import pandas as pd
# from hm import file_handling
# from hm import disclaimer
# from hm.DeterministicRunner import DeterministicRunner
# from hm.ModelTime import ModelTime
# from hm.Configuration import Configuration

from hm import disclaimer
from hm.dynamicmodel import HmDynamicModel
from hm.dynamicframework import HmDynamicFramework
from hm.modeltime import ModelTime
from hm.config import Configuration
from hm.utils import *
from hm.api import set_modeltime, set_domain

from .etrefconfig import ETRefConfiguration
from .penmanmonteith import PenmanMonteith
from .hargreaves import Hargreaves
from .priestleytaylor import PriestleyTaylor
from . import variable_list

import logging
logger = logging.getLogger(__name__)


@click.command()
@click.option('--debug/--no-debug', default=False)
@click.option('-o', '--outputdir', 'outputdir', default='.', type=click.Path())
@click.argument('config', type=click.Path(exists=True))
def cli(debug, outputdir, config):

    # load configuration
    configuration = ETRefConfiguration(
        config,
        outputdir,
        debug
    )

    # create modeltime object
    print(type(configuration.CLOCK['start_time']))
    modeltime = set_modeltime(
        pd.Timestamp(configuration.CLOCK['start_time']),
        pd.Timestamp(configuration.CLOCK['end_time']),
        pd.Timedelta('1 day')
    )

    # set model domain
    domain = set_domain(
        configuration.MODEL_GRID['mask'],
        modeltime,
        configuration.MODEL_GRID['mask_varname'],
        configuration.MODEL_GRID['area_varname'],
        configuration.MODEL_GRID['is_1d'],
        configuration.MODEL_GRID['xy_dimname']# ,
        # z_coords=None,
        # pseudo_coords=configuration.PSEUDO_COORDS
    )

    def create_dynamic_model(method):
        return HmDynamicModel(
            method,
            configuration,
            modeltime,
            domain,
            variable_list,
            initial_state
            )
    
    initial_state=None
    etref_method = configuration.ET_METHOD['method']
    if 'PenmanMonteith' in etref_method:
        dynamic_model = create_dynamic_model(PenmanMonteith)
        # dynamic_model = HmDynamicModel(
        #     PenmanMonteith,
        #     configuration,
        #     modeltime,
        #     domain,
        #     variable_list,
        #     initial_state
        # )        
    if 'Hargreaves' in etref_method:
        dynamic_model = create_dynamic_model(Hargreaves)        
    if 'PriestleyTaylor' in etref_method:
        dynamic_model = create_dynamic_model(PriestleyTaylor)
    
    # run model
    dynamic_framework = HmDynamicFramework(dynamic_model, len(modeltime) + 1)
    dynamic_framework.setQuiet(True)
    dynamic_framework.run()
    

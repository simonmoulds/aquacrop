#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

def run(method, config, modeltime, domain, init):
    dynamic_model = HmDynamicModel(
        method,
        config,
        modeltime,
        domain,
        variable_list,
        init
    )
    dynamic_framework = HmDynamicFramework(dynamic_model, len(modeltime))
    dynamic_framework.setQuiet(True)
    dynamic_framework.run()

# TODO: can this be a class, rather than a function?
def main():

    disclaimer.print_disclaimer()
    iniFileName = os.path.abspath(sys.argv[1])
    debug_mode = False
    if len(sys.argv) > 2:
        if sys.argv[2] == 'debug':
            debug_mode = True

    # configuration = Configuration(
    #     iniFileName=iniFileName,
    #     debug_mode=debug_mode)
    configuration = ETRefConfiguration(
        config_filename=iniFileName,
        debug_mode=debug_mode
    )
    modeltime = ModelTime(
        pd.Timestamp(configuration.CLOCK['startTime']),
        pd.Timestamp(configuration.CLOCK['endTime']),
        pd.Timedelta(configuration.CLOCK['timeDelta'])
    )
    domain = set_domain(
        configuration.MODEL_GRID['mask'],
        modeltime,
        configuration.MODEL_GRID['mask_varname'],
        configuration.MODEL_GRID['area_varname'],
        configuration.MODEL_GRID['is_1d'],
        configuration.MODEL_GRID['xy_dimname'],
        configuration.PSEUDO_COORDS
    )    
    # currTimeStep = ModelTime()
    # initial_state = None
    # currTimeStep.getStartEndTimeSteps(
    #     configuration.CLOCK['startTime'],
    #     configuration.CLOCK['endTime'])
    logger.info('Transient simulation run has started')

    initial_state = None
    etref_method = "Hargreaves"
    # etref_method = 'PenmanMonteith'
    # etref_method = configuration.RefETMethodSelection
    if 'PenmanMonteith' in etref_method:
        run(PenmanMonteith, configuration, modeltime, domain, initial_state)#, 'PenmanMonteith')
    if 'Hargreaves' in etref_method:
        run(Hargreaves, configuration, modeltime, domain, initial_state)#, 'Hargreaves')
    if 'PriestleyTaylor' in etref_method:
        run(PriestleyTaylor, configuration, modeltime, domain, initial_state)#, 'PriestleyTaylor')
    clear_cache()
    
if __name__ == '__main__':
    disclaimer.print_disclaimer(with_logger=True)
    sys.exit(main())

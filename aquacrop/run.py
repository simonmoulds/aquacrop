#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from hm.dynamicFramework import DynamicFramework
from hm.DeterministicRunner import DeterministicRunner
from hm.ModelTime import ModelTime
from hm import disclaimer
from hm.Reporting import Reporting

from hm import file_handling
from .WaterBalanceModel import WaterBalanceModel
from .AquaCropConfiguration import AquaCropConfiguration
from . import variable_list

import logging
logger = logging.getLogger(__name__)

def main(argv):

    # disclaimer.print_disclaimer()

    # get the full path of the config file provided as system argument
    iniFileName = os.path.abspath(argv[0])
    debug_mode = False
    if len(argv) > 2:
        if (argv[0] == "debug"):
            debug_mode = True

    configuration = AquaCropConfiguration(iniFileName=iniFileName, debug_mode=debug_mode)
    currTimeStep = ModelTime()
    initial_state = None
    currTimeStep.getStartEndTimeSteps(
        configuration.CLOCK['startTime'],
        configuration.CLOCK['endTime'])
    currTimeStep.update(1)    
    logger.info('Transient simulation run has started')
    deterministic_runner = DeterministicRunner(
        WaterBalanceModel,
        configuration,
        currTimeStep,
        variable_list,
        initial_state)
    dynamic_framework = DynamicFramework(deterministic_runner, currTimeStep.nrOfTimeSteps)
    dynamic_framework.setQuiet(True)
    dynamic_framework.run()
    file_handling.clear_cache()

if __name__ == '__main__':
    disclaimer.print_disclaimer(with_logger = True)
    sys.exit(main(sys.argv[1:]))

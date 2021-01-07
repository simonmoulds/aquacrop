#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

import os
import sys

from hm import disclaimer
from hm.dynamicmodel import HmDynamicModel, HmMonteCarloModel
from hm.dynamicframework import HmDynamicFramework, HmMonteCarloFramework, HmEnsKalmanFilterFramework
from hm.config import Configuration
from hm.utils import *
from hm.api import set_modeltime, set_domain
# TODO: create api for HmDynamicModel, HmDynamicFramework

# from .enkfmodel import AqEnKfModel
from ..aquacrop.AquaCrop import AqEnKfModel
from ..aquacrop.AquaCrop import AquaCrop
from ..aquacrop.io.AquaCropConfiguration import AquaCropConfiguration
from ..aquacrop.io import variable_list_crop

from ..etref.penmanmonteith import PenmanMonteith
from ..etref.hargreaves import Hargreaves
from ..etref.priestleytaylor import PriestleyTaylor
from ..etref import variable_list

import logging
logger = logging.getLogger(__name__)

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

def run_etref(method, config, modeltime, domain, init):
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
    output_fn = dynamic_model.model.reporting_module.output_variables['ETref_daily_total'].filename
    return output_fn

@click.command()
@click.option('--debug/--no-debug', 'debug', default=False)
@click.option('-o', '--outputdir', 'outputdir', default='.', type=click.Path())
@click.option('--monte-carlo', 'montecarlo', default=False)
@click.option('--enkf', 'kalmanfilter', default=False)
@click.argument('config', type=click.Path(exists=True))
def cli(debug, outputdir, montecarlo, kalmanfilter, config):
    """Example script"""

    # load configuration
    configuration = AquaCropConfiguration(
        config,
        outputdir,
        debug
    )

    # create modeltime object
    modeltime = set_modeltime(
        pd.Timestamp(configuration.CLOCK['start_time']),
        pd.Timestamp(configuration.CLOCK['end_time']),
        pd.Timedelta('1 day')
    )

    # retrieve z coordinate information from config
    dz_lyr = configuration.SOIL_PROFILE['dzLayer']
    z_lyr_bot = np.cumsum(dz_lyr)
    z_lyr_top = z_lyr_bot - dz_lyr
    z_lyr_mid = (z_lyr_top + z_lyr_bot) / 2
    dz_comp = configuration.SOIL_PROFILE['dzComp']
    z_comp_bot = np.cumsum(dz_comp)
    z_comp_top = z_comp_bot - dz_comp
    z_comp_mid = (z_comp_top + z_comp_bot) / 2
    z_coords = {
        'layer': z_lyr_mid,
        'depth': z_comp_mid
    }

    # set model domain
    # settings_dir = os.path.split(os.path.abspath(config))[0]
    domain = set_domain(
        # os.path.join(workdir, configuration.MODEL_GRID['mask']),
        configuration.MODEL_GRID['mask'],
        modeltime,
        configuration.MODEL_GRID['mask_varname'],
        configuration.MODEL_GRID['area_varname'],
        configuration.MODEL_GRID['is_1d'],
        configuration.MODEL_GRID['xy_dimname'],
        z_coords,
        configuration.PSEUDO_COORDS
    )
    # decide whether to preprocess etref
    # would this be better as an option?
    if configuration.ETREF['preprocess']:
        initial_state = None
        etref_method = configuration.ETREF['method']
        if 'PenmanMonteith' in etref_method:
            etref_fn = run_etref(PenmanMonteith, configuration, modeltime, domain, initial_state)
        if 'Hargreaves' in etref_method:
            etref_fn = run_etref(Hargreaves, configuration, modeltime, domain, initial_state)
        if 'PriestleyTaylor' in etref_method:
            etref_fn = run_etref(PriestleyTaylor, configuration, modeltime, domain, initial_state)
            
        # Update configuration
        configuration.ETREF['filename'] = etref_fn
        configuration.ETREF['varname'] = 'etref'
        if domain.is_1d:
            configuration.ETREF['is_1d'] = True
            configuration.ETREF['xy_dimname'] = 'space'            
        clear_cache()

    # run model using user-specified model framework
    if kalmanfilter:
        initial_state = None
        modeltime.reset()
        dynamic_model = AqEnKfModel(
            AquaCrop,
            configuration,
            modeltime,
            domain,
            variable_list_crop,
            initial_state
        )
        dynamic_framework = HmDynamicFramework(
            dynamic_model,
            lastTimeStep=len(modeltime) + 1,
            firstTimestep=1
        )
        dynamic_framework.setQuiet(True)        
        mc_framework = HmMonteCarloFramework(dynamic_framework, nrSamples=5)
        enkf_framework = HmEnsKalmanFilterFramework(mc_framework)
        enkf_framework.setFilterTimesteps([240, 250, 260, 270])
        enkf_framework.run()
        
    elif montecarlo:
        initial_state = None
        modeltime.reset()
        dynamic_model = HmMonteCarloModel(
            AquaCrop,
            configuration,
            modeltime,
            domain,
            variable_list_crop,
            initial_state
        )
        dynamic_framework = HmDynamicFramework(
            dynamic_model,
            lastTimeStep=len(modeltime) + 1,
            firstTimestep=1
        )
        dynamic_framework.setQuiet(True)
        mc_framework = HmMonteCarloFramework(dynamic_framework, nrSamples=5)
        mc_framework.run()
        
    else:
        initial_state = None
        modeltime.reset()
        dynamic_model = HmDynamicModel(
            AquaCrop,
            configuration,
            modeltime,
            domain,
            variable_list_crop,
            initial_state
        )
        dynamic_framework = HmDynamicFramework(
            dynamic_model,
            lastTimeStep=len(modeltime) + 1,
            firstTimestep=1
        )
        dynamic_framework.setQuiet(True)
        dynamic_framework.run()

    
    

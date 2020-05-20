import click

import os
import sys

from hm import disclaimer
from hm.dynamicmodel import HmDynamicModel
from hm.dynamicframework import HmDynamicFramework
from hm.config import Configuration
from hm.utils import *
from hm.api import set_modeltime, set_domain
# TODO: create api for HmDynamicModel, HmDynamicFramework

from ..AquaCrop import AquaCrop
from ..io.AquaCropConfiguration import AquaCropConfiguration
from ..io import variable_list

import logging
logger = logging.getLogger(__name__)


@click.command()
@click.option('--debug/--no-debug', default=False)
@click.option('-o', '--outputdir', 'outputdir', default='.', type=click.Path())
@click.argument('config', type=click.Path(exists=True))
def cli(debug, outputdir, config):
    """Example script"""

    # load configuration
    configuration = AquaCropConfiguration(
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
    # z_coords = {
    #     'layer': z_lyr_mid,
    #     'depth': z_comp_mid
    # }

    # set model domain
    domain = set_domain(
        configuration.MODEL_GRID['mask'],
        modeltime,
        configuration.MODEL_GRID['mask_varname'],
        configuration.MODEL_GRID['area_varname'],
        configuration.MODEL_GRID['is_1d'],
        configuration.MODEL_GRID['xy_dimname'],
        z_coords,
        configuration.PSEUDO_COORDS
    )

    # create dynamic model object
    initial_state = None
    dynamic_model = HmDynamicModel(
        AquaCrop,
        configuration,
        modeltime,
        domain,
        variable_list,
        initial_state
    )
    # run model
    dynamic_framework = HmDynamicFramework(dynamic_model, len(modeltime) + 1)
    dynamic_framework.setQuiet(True)
    dynamic_framework.run()

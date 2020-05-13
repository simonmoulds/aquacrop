Soil Parameters
===============

.. currentmodule:: aquacrop

Configuration
~~~~~~~~~~~~~

Example
^^^^^^^

.. code-block:: python

    [SOIL_PARAMETERS]
    filename = soil_params.nc
    adjust_raw = false
    adjust_cn = true

Options
^^^^^^^

``filename``
    NetCDF file containing soil parameters.

``adjust_raw``
    TODO

``adjust_cn``
    TODO

In addition to these options, users may supply any parameter values, or none, in
the configuration file directly. 

Parameters
~~~~~~~~~~

+--------------+-----------------------------+---------+-----------------------+
| Parameter    | Description                 | Units   |         Default       |
+==============+=============================+=========+=======================+
| z_soil       | Total thickness of the      | m       | \-                    |
|              | soil profile                |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| evap_z_surf  | Thickness of the evaporating| m       | \-                    |
|              | soil surface layer in direct|         |                       |
|              | contact with the atmosphere |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| evap_z_min   | Minimum thickness of soil   | m       | 0.15                  |
|              | surface evaporation layer   |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| evap_z_max   | Maximum thickness of soil   | m       | 0.30                  |
|              | surface evaporation layer   |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| kex          | Maximum soil evaporation    | \-      | 1.10                  |
|              | coefficient                 |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| fevap        | Shape factor which controls | \-      | 4                     |
|              | the reduction in            |         |                       | 
|              | evaporation                 |         |                       |
|              | with decreasing water       |         |                       |
|              | content in the soil surface |         |                       |
|              | layer                       |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| f_wrel_exp   | Water content of the soil   | \-      | 0.40                  |
|              | soil surface layer at       |         |                       |
|              | which the evaporation depth |         |                       |
|              | expands                     |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| fwcc         | Coefficient for the         | \-      | 0.5                   |
|              | reduction in soil           |         |                       |
|              | evaporation due to          |         |                       |
|              | sheltering from withered    |         |                       |
|              | canopy cover                |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| adj_rew      | Switch to use               | \-      | false                 |
|              | a user-defined              |         |                       |
|              | value of readily evaporable |         |                       |
|              | water in place of the       |         |                       |
|              | in place of the calculated  |         |                       |
|              | value                       |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| rew          | User-defined readily        | m       | \-                    |
|              | evaporable water            |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| adj_cn       | Switch to adjust the curve  | \-      | false                 |
|              | number based on the daily   |         |                       |
|              | variation of soil moisture  |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| cn           | Curve number                | \-      | \-                    |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| z_cn         | Thickness of surface        | m       | 0.30                  |
|              | soil layer                  |         |                       |
|              | over which to compute the   |         |                       |
|              | soil moisture content used  |         |                       |
|              | to adjust the curve number, |         |                       |
|              | if ``adj_cn`` is ``true``   |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| z_germ       | Thickness of surface        | m       | 0.30                  |
|              | soil layer over which to    |         |                       |
|              | compute the soil moisture   |         |                       |
|              | content used to determine   |         |                       |
|              | whether germination can     |         |                       |
|              | occur                       |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| z_res        | Depth to soil layer         | m       | Use negative value    |
|              | which restricts root growth |         | if no restrictive     |
|              |                             |         | layer is present      |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| fshape_cr    | Shape factor controlling    | \-      | 16                    |
|              | the effect of shallow       |         |                       |
|              | groundwater on soil water   |         |                       |
|              | content                     |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+

Irrigation Parameters
=====================

.. currentmodule:: aquacrop

Configuration
~~~~~~~~~~~~~

.. code-block:: python

    [IRRIGATION_PARAMETERS]
    filename = irr_params.nc
    
Options
~~~~~~~

``filename``
    NetCDF file containing irrigation parameter values.

Parameters
~~~~~~~~~~

+--------------+-----------------------------+---------+-----------------------+
| Parameter    | Description                 | Units   |      Default          |
+==============+=============================+=========+=======================+
| irr_method   | | Method of irrigation:     | \-      | 0 (Rainfed)           |
|              | | 0 = Rainfed               |         |                       |
|              | | 1 = Soil moisture         |         |                       |
|              | | 2 = Fixed interval        |         |                       |
|              | | 3 = Pre-specified depths  |         |                       |
|              | | 4 = Net irrigation        |         |                       | 
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| irr_interval | Time between irrigation     | Days    |                       |
|              | events. Only used if        |         |                       |
|              | irr_method = 2.             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| smt1         | Percentage of total         | \%      |                       |
|              | available water during the  |         |                       |
|              | first crop growth stage     |         |                       |
|              | below which irrigation is   |         |                       |
|              | initiated. Only used if     |         |                       |
|              | irr_method = 1.             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| smt2         | As above, but for the       | \%      |                       |
|              | second crop growth stage    |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| smt3         | As above, but for the       | \%      |                       |
|              | third crop growth stage.    |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| smt4         | As above, but for the       | \%      |                       |
|              | fourth crop growth stage.   |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| max_irr      | Maximum irrigation depth.   | mm/day  |                       |
|              |                             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| app_eff      | Irrigation application      | \%      |                       |
|              | efficiency.                 |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| net_irr_smt  | Percentage of total         | \%      |                       |
|              | available water to maintain.|         |                       |
|              | during growing season. Only |         |                       |
|              | used if irr_method = 4.     |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| wet_surf     | Percentage of soil surface  | \%      |                       | 
|              | area wetted by irrigation.  |         |                       | 
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+


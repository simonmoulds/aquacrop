Soil Hydraulic Parameters
=========================

.. currentmodule:: aquacrop

Configuration
~~~~~~~~~~~~~

.. code-block:: python

    [SOIL_HYDRAULIC_PARAMETERS]
    filename = "aquacrop_soil_parms_pt.nc"
    k_sat_varname = "k_sat"
    th_sat_varname = "th_sat"
    th_fc_varname = "th_fc"
    th_wilt_varname = "th_wilt"
    is_1d = true
    xy_dimname = "space"

Options
~~~~~~~

SOIL_HYDRAULIC_PARAMETERS
^^^^^^^^^^^^^^^^^^^^^^^^^

``filename``
    Path to netCDF file containing soil hydraulic parameters
    
``k_sat_varname``
    Name of saturated hydraulic_conductivity variable in file.
    
``th_sat_varname``
    Name of saturated water content variable in file.
    
``th_fc_varname``
    Name of field_capacity variable in file.
    
``th_wilt_varname``
    Name of wilting point variable in file.
    
``is_1d``
    Is the file one-dimensional?
    
``xy_dimname``
    If ``is_1d = true``, then this option defines the name of the space dimension in the file.

In addition to these options, users may supply any parameter values, or none, in the configuration file directly. When reading these parameter values, the program initially checks the configuration file, then the supplied netCDF file. If a single value is provided then this is broadcast to each soil layer defined by ``dzLayer``. Otherwise an array may be supplied, which must be the same size as ``dzLayer``. 

Parameters
~~~~~~~~~~

+--------------+-----------------------------+---------+-----------------------+
| Parameter    | Description                 | Units   |      Default          |
+==============+=============================+=========+=======================+
| th_fc        | Field capacity.             | m3/m3   | \-                    |
|              |                             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| th_sat       | Saturated water content.    | m3/m3   | \-                    |
|              |                             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| th_wilt      | Wilting point.              | m3/m3   | \-                    |
|              |                             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| k_sat        | Saturated hydraulic         | mm/day  | \-                    |
|              | conductivity.               |         |                       |
|              |                             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+

    

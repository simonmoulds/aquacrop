Soil Hydraulic Parameters
=========================

.. currentmodule:: aquacrop

Configuration
~~~~~~~~~~~~~

.. code-block:: python

    [SOIL_HYDRAULIC_PARAMETERS]
    dzLayer = 1, 2, 3
    filename = field_params.nc

Options
~~~~~~~

SOIL_HYDRAULIC_PARAMETERS
^^^^^^^^^^^^^^^^^^^^^^^^^

``dzLayer``
    Thickness of each layer in the soil profile
``filename``
    Path to netCDF file containing soil hydraulic parameters

Parameters
~~~~~~~~~~

+--------------+-----------------------------+---------+-----------------------+
| Parameter    | Description                 | Units   |      Default          |
+==============+=============================+=========+=======================+
| th_fc        | Crop category               | m3 m-3  | \-                    |
|              |                             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| th_sat       |                             | m3 m-3  | \-                    |
|              |                             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| th_wilt      |                             | m3 m-3  | \-                    |
|              |                             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+

    
SOIL_PROFILE
^^^^^^^^^^^^

``dzLayer``
    TODO

``dzComp``
    TODO

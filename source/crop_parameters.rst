Crop Parameters
===============

.. currentmodule:: aquacrop

Configuration
~~~~~~~~~~~~~

.. code-block:: python

    [CROP_PARAMETERS]
    calendar_type = 1
    switch_gdd = true
    gdd_method = 1
    filename = crop_params.nc

Options
~~~~~~~

``calendar_type``
    TODO

``switch_gdd``
    TODO

``gdd_method``
    TODO

``filename``
    NetCDF file containing crop parameters.

Parameters
~~~~~~~~~~

+--------------+----------------------------+---------+-----------------------+
| Parameter    | Description                | Units   |         Default       |
+==============+============================+=========+=======================+
|              |                            |         |                       |
| CropType     |  Crop category             | \-      | | 1 = Leafy vegetable |
|              |                            |         | | 2 = Root/tuber      |
|	       |		            |	      | | 3 = Fruit/grain     |
|              |                            |         |                       |
+--------------+----------------------------+---------+-----------------------+
| CalendarType |	                    | \-      | 1                     |
+--------------+----------------------------+---------+-----------------------+

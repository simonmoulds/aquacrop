Field Parameters
================

.. currentmodule:: aquacrop

Configuration
~~~~~~~~~~~~~

.. code-block:: python

    [FIELD_PARAMETERS]
    filename = field_params.nc

Options
~~~~~~~

``filename``
    NetCDF file containing soil parameters.

Parameters
~~~~~~~~~~

+--------------+-----------------------------+---------+-----------------------+
| Parameter    | Description                 | Units   |      Default          |
+==============+=============================+=========+=======================+
| mulches      | | Is the soil surface       |         | 0 (No)                |
|              | | covered by mulches?       |         |                       |
|              | | 0 = No                    |         |                       |
|              | | 1 = Yes                   |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| bunds        | | Are soil bunds present?   |         | 0 (No)                |
|              | | 0 = No                    |         |                       |
|              | | 1 = Yes                   |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| mulch_pct_gs | Percentage of soil surface  | \%      |                       |
|              | covered by mulches during   |         |                       |
|              | the growing season.         |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| mulch_pct_os | Percentage of soil surface  | \%      |                       |
|              | covered by mulches outside  |         |                       |
|              | of the growing season.      |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| f_mulch      | Factor which defines the    | \-      |                       |
|              | proportional reduction of   |         |                       |
|              | soil evaporation due to the |         |                       |
|              | presence of mulches.        |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| z_bund       | Height of soil bunds.       | Metres  | 0                     |
|              |                             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| bund_water   | Initial depth of water held | Metres  | 0                     |
|              | by soil bunds.              |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+

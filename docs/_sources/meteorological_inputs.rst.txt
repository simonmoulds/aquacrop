Meteorological Inputs
=====================

.. currentmodule:: aquacrop

Configuration
~~~~~~~~~~~~~

.. code-block:: python

   [PRECIPITATION]
   filename = prec.nc
   varname = prec

   [TAVG]
   filename = tavg.nc
   varname = tavg

   [TMIN]
   filename = tmin.nc
   varname = tmin

   [TMAX]
   filename = tmax.nc
   varname = tmax

   [ETREF]
   filename = etref.nc
   varname = etref
   
   [CARBON_DIOXIDE]
   filename = co2.nc
   varname = conc

Options
~~~~~~~

``filename`` : ``str``
    Path to netCDF file containing the meteorological input variable.

``varname`` : ``str``
    NetCDF variable name.




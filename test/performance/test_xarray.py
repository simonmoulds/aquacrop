#!/usr/bin/env python
# -*- coding: utf-8 -*-

import netCDF4 as nc
import xarray as xr
import numpy as np
import timeit

def compute_extraterrestrial_radiation(latitudes, julian_day):
    LatRad = latitudes * np.pi / 180.0
    declin = (0.4093 * (np.sin(((2.0 * np.pi * julian_day) / 365.0) - 1.405)))
    arccosInput = (-(np.tan(LatRad)) * (np.tan(declin)))
    arccosInput = np.clip(arccosInput, -1, 1)
    sunangle = np.arccos(arccosInput)
    distsun = 1 + 0.033 * (np.cos((2 * np.pi * julian_day) / 365.0))
    extraterrestrial_radiation = (
        ((24 * 60 * 0.082) / np.pi)
        * distsun
        * (sunangle
           * (np.sin(LatRad))
           * (np.sin(declin))
           + (np.cos(LatRad))
           * (np.cos(declin))
           * (np.sin(sunangle)))) # MJ m-2 d-1
    return extraterrestrial_radiation

temp_fn = "/home/simon/projects/AquaCrop/Input/temp_aos_default_data.nc"

# xarray
start_time = timeit.default_timer()
temp_ds = xr.open_dataset(temp_fn)#, chunks={'time' : 1})
nt = 1000#len(temp_ds.time)
for i in range(nt):
    tmin = temp_ds.Tmin.data[i,...]
    tmax = temp_ds.Tmax.data[i,...]
    tmean = (tmin + tmax) / 2
    latitudes = temp_ds.latitude.data[:,None] * np.ones((temp_ds.longitude.size))[None,:]
    doy = temp_ds.time.dt.dayofyear.data[i]
    etrad = compute_extraterrestrial_radiation(latitudes, doy)
    etref = (0.0023
             * (etrad * 0.408)  # MJ m-2 d-1 -> mm d-1
            * ((np.maximum(0, (tmean - 273.0))) + 17.8)
            * np.sqrt(np.maximum(0,(tmax - tmin))))
time_elapsed = timeit.default_timer() - start_time
print(time_elapsed)

# netcdf
start_time = timeit.default_timer()
temp_ds = nc.Dataset(temp_fn)
t_unit = temp_ds.variables['time'].units
t_cal = temp_ds.variables['time'].calendar
time = nc.num2date(temp_ds.variables['time'][:], units=t_unit, calendar=t_cal)
nt = 1000#temp_ds.variables['time'].size
for i in range(nt):
    tmin = temp_ds.variables['Tmin'][i,...]
    tmax = temp_ds.variables['Tmax'][i,...]
    tmean = (tmin + tmax) / 2
    latitudes = temp_ds.variables['latitude'][:][:,None] * np.ones((temp_ds.variables['longitude'].size))[None,:]
    doy = time[i].timetuple().tm_yday
    etrad = compute_extraterrestrial_radiation(latitudes, doy)
    etref = (0.0023
             * (etrad * 0.408)  # MJ m-2 d-1 -> mm d-1
            * ((np.maximum(0, (tmean - 273.0))) + 17.8)
            * np.sqrt(np.maximum(0,(tmax - tmin))))
time_elapsed = timeit.default_timer() - start_time
print(time_elapsed)

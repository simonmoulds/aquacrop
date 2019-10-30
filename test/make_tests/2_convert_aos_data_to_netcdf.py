#!/usr/bin/env python3

# Python script to convert AquaCropOS input files to netCDF
# for AquaCrop_Py

import os
import numpy as np
import datetime
from netCDF4 import Dataset, date2num
from pandas import read_csv
import scipy.interpolate as interpolate
import re
import rasterio
import time
import shutil

matlabdir = 'aquacrop_matlab_test_data'
pythondir = 'aquacrop_python_test_data'

# clean directory to store output
if os.path.exists(pythondir):
    shutil.rmtree(pythondir)
    
os.mkdir(pythondir)

# tests = {
#     'Ch7_Ex1a_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex1a_tunis_wheat',
#         'config_years' : range(1979,2002)}
#     }

tests = {
    'Ghana_AgMERRA' : {
        'nc_prefix'    : 'aos_ghana_agmerra_maize',
        'config_years' : range(2007,2008)}
    }

# tests = {
#     'Ch7_Ex1a_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex1a_tunis_wheat',
#         'config_years' : list(range(1979,2002))},
#     'Ch7_Ex1b_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex1b_tunis_wheat',
#         'config_years' : list(range(1979,2002))},
#     'Ch7_Ex2a_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex2a_tunis_wheat',
#         'config_years' : list(range(1979,2002))},
#     'Ch7_Ex2b_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex2b_tunis_wheat',
#         'config_years' : list(range(1979,2002))},
#     'Ch7_Ex3a_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex3a_tunis_wheat',
#         'config_years' : list(range(1988,1989))},
#     'Ch7_Ex3b_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex3b_tunis_wheat',
#         'config_years' : list(range(1988,1989))},
#     'Ch7_Ex3c_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex3c_tunis_wheat',
#         'config_years' : list(range(1988,1989))},
#     'Ch7_Ex3d_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex3d_tunis_wheat',
#         'config_years' : list(range(1988,1989))},
#     'Ch7_Ex6_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex6_tunis_wheat',
#         'config_years' : list(range(1979,2002))},
#     'Ch7_Ex7a_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex7a_tunis_wheat',
#         'config_years' : list(range(1979,2002))},
#     'Ch7_Ex7b_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex7b_tunis_wheat',
#         'config_years' : list(range(1979,2002))},
#     'Ch7_Ex7c_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex7c_tunis_wheat',
#         'config_years' : list(range(1979,2002))},
#     'Ch7_Ex7d_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex7d_tunis_wheat',
#         'config_years' : list(range(1979,2002))},
#     'Ch7_Ex7e_Tunis_Wheat' : {
#         'nc_prefix'    : 'aos_ch7_ex7e_tunis_wheat',
#         'config_years' : list(range(1979,2002))},    
#     'Ch8_Ex2a_Hyderabad_Cereal' : {
#         'nc_prefix'    : 'aos_ch8_ex2a_hyderabad_cereal',
#         'config_years' : list(range(2000,2011))},
#     'Ch8_Ex2b_Hyderabad_Cereal' : {
#         'nc_prefix'    : 'aos_ch8_ex2b_hyderabad_cereal',
#         'config_years' : list(range(2000,2011))},
#     'Ch8_Ex3a_Hyderabad_Cereal' : {
#         'nc_prefix'    : 'aos_ch8_ex3a_hyderabad_cereal',
#         'config_years' : list(range(2002,2003))},
#     'Ch8_Ex3b_Hyderabad_Cereal' : {
#         'nc_prefix'    : 'aos_ch8_ex3b_hyderabad_cereal',
#         'config_years' : list(range(2002,2003))},
#     'Ch8_Ex3c_Hyderabad_Cereal' : {
#         'nc_prefix'    : 'aos_ch8_ex3c_hyderabad_cereal',
#         'config_years' : list(range(2002,2003))},
#     'Ch8_Ex3d_Hyderabad_Cereal' : {
#         'nc_prefix'    : 'aos_ch8_ex3d_hyderabad_cereal',
#         'config_years' : list(range(2002,2003))},        
#     'Ch8_Ex6_Hyderabad_Cereal' : {
#         'nc_prefix'    : 'aos_ch8_ex6_hyderabad_cereal',
#         'config_years' : list(range(2000,2010))},        
#     'Ch9_Ex1_Brussels_Potato'  : {
#         'nc_prefix'    : 'aos_ch9_ex1_brussels_potato',
#         'config_years' : list(range(1985,2006))},
#     'Ch9_Ex4_Brussels_Potato'  : {
#         'nc_prefix'    : 'aos_ch9_ex4_brussels_potato',
#         'config_years' : list(range(1985,2006))},
#     'Ch9_Ex5a_Brussels_Potato' : {
#         'nc_prefix'    : 'aos_ch9_ex5a_brussels_potato',
#         'config_years' : list(range(1985,2006))},
#     'Ch9_Ex5b_Brussels_Potato' : {
#         'nc_prefix'    : 'aos_ch9_ex5b_brussels_potato',
#         'config_years' : list(range(1985,2006))},
#     'Ch9_Ex5c_Brussels_Potato' : {
#         'nc_prefix'    : 'aos_ch9_ex5c_brussels_potato',
#         'config_years' : list(range(1985,2006))},
#     'Ch9_Ex6a_Brussels_Potato' : {
#         'nc_prefix'    : 'aos_ch9_ex6a_brussels_potato',
#         'config_years' : list(range(1976,2006))},
#     'Ch9_Ex6b_Brussels_Potato' : {
#         'nc_prefix'    : 'aos_ch9_ex6b_brussels_potato',
#         'config_years' : list(range(2041,2071))}
# }

def run():
    for test in list(tests.keys()):
        testdir = str(test)
        print(testdir)
        # inputdir = tests[test]['inputdir']
        # outputdir = tests[test]['outputdir']
        nc_prefix = tests[test]['nc_prefix']
        config_years = tests[test]['config_years']

        os.mkdir(os.path.join(pythondir, testdir))

        for yr in config_years:
            print(yr)

            os.mkdir(os.path.join(pythondir, testdir, str(yr)))
            inputdir = os.path.join(testdir, str(yr), 'Input')
            outputdir = os.path.join(testdir, str(yr), 'Output')
            configdir = os.path.join(testdir, str(yr), 'Config')
            os.mkdir(os.path.join(pythondir, inputdir))
            os.mkdir(os.path.join(pythondir, outputdir))
            os.mkdir(os.path.join(pythondir, configdir))

            # crop mix data
            # =============
            crop_mix_fn = os.path.join(
                os.path.join(matlabdir,inputdir),
                'CropMix.txt')
            with open(crop_mix_fn) as f:
                content = f.read().splitlines()

            # the following command to remove commented lines
            content = [x for x in content if re.search('^(?!%%).*', x)]
            ncrop = int(content[0])

            # soil profile data
            # =================
            soil_profile = read_csv(
                os.path.join(os.path.join(matlabdir,inputdir),
                             'SoilProfile.txt'),
                delimiter='\s+|\t',
                header=None,
                names=['compartment','thickness','layer'],
                skiprows=2,
                engine='python')

            zcomp = np.array(soil_profile['thickness'])
            zbot = np.cumsum(zcomp)
            ztop = zbot - zcomp
            zmid = (zbot + ztop) / 2
            ncompartment = len(soil_profile['compartment'].values)

            # soil hydrology data
            # ===================
            soil_hydrology = read_csv(
                os.path.join(os.path.join(matlabdir,inputdir),
                             'SoilHydrology.txt'),
                delimiter='\s+|\t',
                header=None,
                names=['layer','thickness','th_s','th_fc','th_wp','ksat'],
                skiprows=2,
                engine='python')

            nlayer = np.max(soil_hydrology['layer'].values)
            zLayer = np.array(soil_hydrology['thickness'].values)
            zLayerBot = np.cumsum(zLayer)
            zLayerTop = zLayerBot - zLayer
            zLayerMid = (zLayerTop + zLayerBot) / 2
            th_s = soil_hydrology['th_s'].values 
            th_fc = soil_hydrology['th_fc'].values
            th_wp = soil_hydrology['th_wp'].values
            k_sat = soil_hydrology['ksat'].values

            # crop parameters
            # ===============
            def read_params(fn):
                with open(fn) as f:
                    content = f.read().splitlines()

                # remove commented lines
                content = [x for x in content if re.search('^(?!%%).*', x)]
                content = [re.split('\s*:\s*', x) for x in content]
                params = {}
                for x in content:
                    if len(x) > 1:
                        nm = x[0]
                        val = x[1]
                        params[nm] = val
                return params

            crop_params_fn = os.path.join(
                os.path.join(matlabdir,inputdir),
                'Crop.txt')
            crop_params = read_params(crop_params_fn)

            # soil parameters
            # ===============
            soil_params_fn = os.path.join(
                matlabdir,
                inputdir,
                'Soil.txt')
            soil_params = read_params(soil_params_fn)

            # field management parameters
            # ===========================
            mgmt_params_fn = os.path.join(
                os.path.join(matlabdir,inputdir),
                'FieldManagement.txt')
            mgmt_params = read_params(mgmt_params_fn)

            # irrigation management parameters
            # ================================
            irri_params_fn = os.path.join(
                os.path.join(matlabdir,inputdir),
                'IrrigationManagement.txt')
            irri_params = read_params(irri_params_fn)

            # spatial data
            # ============

            # use arbitrary lat/lon vals, because we just want to
            # test the routines
            lat_vals = np.array([26.75, 26.25], dtype='f4')
            lon_vals = np.array([84.25, 84.75], dtype='f4')
            nlat = lat_vals.shape[0]
            nlon = lon_vals.shape[0]
            region_arr = np.ones((nlat,nlon))

            # write region files
            xmin,ymin,xmax,ymax = [lon_vals.min(),lat_vals.min(),lon_vals.max(),lat_vals.max()]
            xres = (xmax-xmin)/(float(nlon) - 1)
            yres = (ymin-ymax)/(float(nlat) - 1)
            # https://rasterio.readthedocs.io/en/stable/quickstart.html#creating-data
            transform = rasterio.transform.Affine.translation(lon_vals[0] - xres/2, lat_vals[0] - yres/2) * rasterio.transform.Affine.scale(xres,yres)
            fill = np.ones((nlat, nlon))
            output_raster = rasterio.open(
                'myraster.tif',
                'w',
                driver='GTiff',
                height=nlat,
                width=nlon,
                count=1,
                dtype=fill.dtype,
                crs='+proj=latlong',
                transform=transform
                )
            output_raster.write(fill, 1)
            output_raster.close()

            os.system('cp myraster.tif test.clone.tif')
            os.system('cp myraster.tif test.landmask.tif')
            os.system('mv test.clone.tif test.landmask.tif ' + os.path.join(pythondir,inputdir))
            os.system('rm myraster.tif')

            # meteorological variables
            # ========================
            weather_fn = os.path.join(
                os.path.join(matlabdir,inputdir),
                'Weather.txt')
            d = read_csv(weather_fn, delimiter="\t", header=1)
            col_nms = d.columns.str.replace('%%','')
            col_nms = col_nms.str.strip()
            d.columns = col_nms

            years = d['Year'].values
            months = d['Month'].values
            days = d['Day'].values

            ntime = len(years)
            dates = []
            for i in range(ntime):
                dates.append(datetime.datetime(years[i], months[i], days[i]))

            prec_vals = (
                d['Precipitation'].values[:,None,None]
                * np.ones((nlat, nlon))[None,:,:])

            tmin_vals = (
                d['MinTemp'].values[:,None,None]
                * np.ones((nlat, nlon))[None,:,:])

            tmax_vals = (
                d['MaxTemp'].values[:,None,None]
                * np.ones((nlat, nlon))[None,:,:])

            etpot_vals = (
                d['ReferenceET'].values[:,None,None]
                * np.ones((nlat, nlon))[None,:,:])

            # Precipitation netCDF
            # ####################
            dataset = Dataset(
                os.path.join(pythondir,
                             inputdir,
                             'prec_' + nc_prefix + '.nc'),
                'w')

            dataset.description = 'AquaCrop meteorological input data'
            dataset.history = 'Created ' + time.ctime(time.time())

            # dataset dimensions
            tm = dataset.createDimension('time', None)
            lat = dataset.createDimension('lat', nlat)
            lon = dataset.createDimension('lon', nlon)

            # dataset variables
            times = dataset.createVariable('time', 'f4', ('time',))
            times.units = "Days since 1901-01-01"
            times.calendar = 'standard'
            times[:] = date2num(
                dates,
                units=times.units,
                calendar=times.calendar)

            latitudes = dataset.createVariable('lat', np.float64, ('lat',))
            latitudes.standard_name = "latitude"
            latitudes.units = "degrees_north"
            latitudes[:] = lat_vals

            longitudes = dataset.createVariable('lon', np.float64, ('lon',))
            longitudes.standard_name = "longitude"
            longitudes.units = "degrees_east"
            longitudes[:] = lon_vals

            prec = dataset.createVariable('precipitation', np.float64, ('time','lat','lon'))
            prec.setncattr('standard_name', 'precipitation')
            # prec.setncattr('long_name', 'precipitation')
            prec.setncattr('units', '1e-3 m.day-1')
            prec.setncattr('_FillValue', np.float64(1e+20))
            prec.setncattr('missing_value', np.float64(1e+20))
            prec[:] = prec_vals

            dataset.close()

            # Temperature netCDF
            # ##################
            dataset = Dataset(
                os.path.join(pythondir,
                             inputdir,
                             'temp_' + nc_prefix + '.nc'),
                'w')
            dataset.description = 'AquaCrop meteorological input data'
            dataset.history = 'Created ' + time.ctime(time.time())

            # dataset dimensions
            tm = dataset.createDimension('time', None)
            lat = dataset.createDimension('lat', nlat)
            lon = dataset.createDimension('lon', nlon)

            # dataset variables
            times = dataset.createVariable('time', 'f4', ('time',))
            times.units = "Days since 1901-01-01"
            times.calendar = 'standard'
            times[:] = date2num(
                dates,
                units=times.units,
                calendar=times.calendar)

            latitudes = dataset.createVariable('lat', np.float64, ('lat',))
            latitudes.standard_name = "latitude"
            latitudes.units = "degrees_north"
            latitudes[:] = lat_vals

            longitudes = dataset.createVariable('lon', np.float64, ('lon',))
            longitudes.standard_name = "longitude"
            longitudes.units = "degrees_east"
            longitudes[:] = lon_vals

            tmin = dataset.createVariable('Tmin', np.float64, ('time','lat','lon'))
            tmin.setncattr('standard_name', 'Tmin')
            tmin.setncattr('long_name', 'minimum_daily_temperature')
            tmin.setncattr('units', 'degrees Celcius')
            tmin.setncattr('_FillValue', np.float64(1e+20))
            tmin.setncattr('missing_value', np.float64(1e+20))
            tmin[:] = tmin_vals

            tmax = dataset.createVariable('Tmax', np.float64, ('time','lat','lon'))
            tmax.setncattr('standard_name', 'Tmax')
            tmax.setncattr('long_name', 'maximum_daily_temperature')
            tmax.setncattr('units', 'degrees Celcius')
            tmax.setncattr('_FillValue', np.float64(1e+20))
            tmax.setncattr('missing_value', np.float64(1e+20))
            tmax[:] = tmax_vals

            dataset.close()

            # Reference ET netCDF
            # ###################
            dataset = Dataset(
                os.path.join(pythondir,
                             inputdir,
                             'eto_' + nc_prefix + '.nc'),
                'w')
            dataset.description = 'AquaCrop meteorological input data'
            dataset.history = 'Created ' + time.ctime(time.time())

            # dataset dimensions
            tm = dataset.createDimension('time', None)
            lat = dataset.createDimension('lat', nlat)
            lon = dataset.createDimension('lon', nlon)

            # dataset variables
            times = dataset.createVariable('time', 'f4', ('time',))
            times.units = "Days since 1901-01-01"
            times.calendar = 'standard'
            times[:] = date2num(dates, units=times.units, calendar=times.calendar)

            latitudes = dataset.createVariable('lat', np.float64, ('lat',))
            latitudes.standard_name = "latitude"
            latitudes.units = "degrees_north"
            latitudes[:] = lat_vals

            longitudes = dataset.createVariable('lon', np.float64, ('lon',))
            longitudes.standard_name = "longitude"
            longitudes.units = "degrees_east"
            longitudes[:] = lon_vals

            etpot = dataset.createVariable('referencePotET', np.float64, ('time','lat','lon'))
            etpot.setncattr('standard_name', 'referencePotET')
            etpot.setncattr('long_name', 'referencePotET')
            etpot.setncattr('units', '1e-3 m.day-1')
            etpot.setncattr('_FillValue', np.float64(1e+20))
            etpot.setncattr('missing_value', np.float64(1e+20))
            etpot[:] = etpot_vals

            dataset.close()

            # parameters
            # ==========
            dataset = Dataset(
                os.path.join(pythondir,
                             inputdir,
                             'params_' + nc_prefix + '.nc'),
                'w')
            dataset.description = 'AquaCrop parameter file'
            dataset.history = 'Created ' + time.ctime(time.time())

            lat = dataset.createDimension('lat', nlat)
            lon = dataset.createDimension('lon', nlon)
            crop = dataset.createDimension('crop', ncrop)
            compartment = dataset.createDimension('compartment', ncompartment)
            layer = dataset.createDimension('layer', nlayer)

            latitudes = dataset.createVariable('lat', np.float64, ('lat',))
            latitudes.standard_name = "latitude"
            latitudes.units = "degrees_north"
            latitudes[:] = lat_vals

            longitudes = dataset.createVariable('lon', np.float64, ('lon',))
            longitudes.standard_name = "longitude"
            longitudes.units = "degrees_east"
            longitudes[:] = lon_vals

            crops = dataset.createVariable('crop', np.int32, ('crop',))
            crops[:] = np.arange(0,ncrop,1)

            compartments = dataset.createVariable('compartment', np.float64, ('compartment',))
            compartments.setncattr('standard_name', 'depth')
            compartments.setncattr('long_name', 'depth below ground level')
            compartments.setncattr('units', 'm')
            compartments[:] = zmid #np.arange(0,ncompartment,1) 

            layers = dataset.createVariable('layer', np.float64, ('layer',))
            layers.setncattr('standard_name', 'depth')
            layers.setncattr('long_name', 'depth below ground level')
            layers.setncattr('units', 'm')
            layers[:] = zLayerMid

            # soil hydraulic properties
            # #########################
            ksat = dataset.createVariable('ksat', np.float64, ('layer','lat','lon'))
            ksat_vals = k_sat[:,None,None] * np.ones((nlat, nlon))[None,:,:]
            ksat[:] = ksat_vals

            wcsat = dataset.createVariable('th_s', np.float64, ('layer','lat','lon'))
            wcsat_vals = th_s[:,None,None] * np.ones((nlat, nlon))[None,:,:]
            wcsat[:] = wcsat_vals

            wp = dataset.createVariable('th_wp', np.float64, ('layer','lat','lon'))
            wp_vals = th_wp[:,None,None] * np.ones((nlat, nlon))[None,:,:]
            wp[:] = wp_vals

            fc = dataset.createVariable('th_fc', np.float64, ('layer','lat','lon'))
            fc_vals = th_fc[:,None,None] * np.ones((nlat, nlon))[None,:,:]
            fc[:] = fc_vals

            # crop parameters
            # ###############

            # TODO: write each parameter individually, with appropriate
            # metadata

            calendar_type = crop_params['CalendarType']
            switch_gdd = crop_params['SwitchGDD']
            gdd_method = crop_params['GDDmethod']

            int_vars = ['CropType','Emergence','MaxRooting','Senescence','Maturity','HIstart','Flowering','YldForm','PolHeatStress','PolColdStress','BioTempStress','PlantPop','Determinant','ETadj','LagAer']

            for var in int_vars:
                a = dataset.createVariable(var, np.int32, ('crop','lat','lon'))
                if var in crop_params:
                    a_vals = np.int32(crop_params[var])
                    a[:] = a_vals * np.ones((ncrop,nlat,nlon))
                else:
                    raise

            float_vars = ['Tbase','Tupp','Tmax_up','Tmax_lo','Tmin_up','Tmin_lo','GDD_up','GDD_lo','fshape_b','PctZmin','Zmin','Zmax','fshape_r','fshape_ex','SxTopQ','SxBotQ','a_Tr','SeedSize','CCmin','CCx','CDC','CGC','Kcb','fage','WP','WPy','fsink','bsted','bface','HI0','HIini','dHI_pre','a_HI','b_HI','dHI0','exc','MaxFlowPct','p_up1','p_up2','p_up3','p_up4','p_lo1','p_lo2','p_lo3','p_lo4','fshape_w1','fshape_w2','fshape_w3','fshape_w4','Aer','beta','GermThr']

            for var in float_vars:
                b = dataset.createVariable(var, np.float64, ('crop','lat','lon'))
                if var in crop_params:
                    b_vals = np.float64(crop_params[var])
                    b[:] = b_vals * np.ones((ncrop,nlat,nlon))

            # get Julian day by assuming some non leap year (e.g. 1999)
            pd = crop_params['PlantingDate'].split('/')
            pd = datetime.datetime(1999,int(pd[1]),int(pd[0])).timetuple().tm_yday
            plantingdate = dataset.createVariable('PlantingDate', np.int32, ('crop','lat','lon'))
            plantingdate[:] = pd

            hd = crop_params['HarvestDate'].split('/')
            hd = datetime.datetime(1999,int(hd[1]),int(hd[0])).timetuple().tm_yday
            harvestdate = dataset.createVariable('HarvestDate', np.int32, ('crop','lat','lon'))
            harvestdate[:] = hd

            # field management parameters
            # ###########################
            mulches = dataset.createVariable('Mulches', np.int32, ('crop','lat','lon'))
            mulches[:] = np.int32(mgmt_params['Mulches'])

            mulchpctgs = dataset.createVariable('MulchPctGS', np.int32, ('crop','lat','lon'))
            mulchpctgs[:] = np.int32(mgmt_params['MulchPctGS'])

            mulchpctos = dataset.createVariable('MulchPctOS', np.int32, ('crop','lat','lon'))
            mulchpctos[:] = np.int32(mgmt_params['MulchPctOS'])

            fmulch = dataset.createVariable('fMulch', np.float64, ('crop','lat','lon'))
            fmulch[:] = np.float64(mgmt_params['fMulch'])

            bunds = dataset.createVariable('Bunds', np.int32, ('crop','lat','lon'))
            bunds[:] = np.int32(mgmt_params['Bunds'])

            zbund = dataset.createVariable('zBund', np.float64, ('crop','lat','lon'))
            zbund[:] = np.float64(mgmt_params['zBund'])

            bundwater = dataset.createVariable('BundWater', np.float64, ('crop','lat','lon'))
            bundwater[:] = np.float64(mgmt_params['BundWater'])

            # irrigation management parameters
            # ################################
            irrmethod = dataset.createVariable('IrrMethod', np.int32, ('crop','lat','lon'))
            irrmethod[:] = np.int32(irri_params['IrrMethod'])

            irrinterval = dataset.createVariable('IrrInterval', np.int32, ('crop','lat','lon'))
            irrinterval[:] = np.int32(irri_params['IrrInterval'])

            smt1 = dataset.createVariable('SMT1', np.float64, ('crop','lat','lon'))
            smt1[:] = np.float64(irri_params['SMT1'])

            smt2 = dataset.createVariable('SMT2', np.float64, ('crop','lat','lon'))
            smt2[:] = np.float64(irri_params['SMT2'])

            smt3 = dataset.createVariable('SMT3', np.float64, ('crop','lat','lon'))
            smt3[:] = np.float64(irri_params['SMT3'])

            smt4 = dataset.createVariable('SMT4', np.float64, ('crop','lat','lon'))
            smt4[:] = np.float64(irri_params['SMT4'])

            maxirr = dataset.createVariable('MaxIrr', np.float64, ('crop','lat','lon'))
            maxirr[:] = np.float64(irri_params['MaxIrr'])

            appeff = dataset.createVariable('AppEff', np.float64, ('crop','lat','lon'))
            appeff[:] = np.float64(irri_params['AppEff'])

            netirrsmt = dataset.createVariable('NetIrrSMT', np.float64, ('crop','lat','lon'))
            netirrsmt[:] = np.float64(irri_params['NetIrrSMT'])

            wetsurf = dataset.createVariable('WetSurf', np.float64, ('crop','lat','lon'))
            wetsurf[:] = np.float64(irri_params['WetSurf'])

            # soil properties
            # ###############
            # calcshp = dataset.createVariable('CalcSHP', np.int32, ('lat','lon'))
            # calcshp[:] = np.int32(soil_params['CalcSHP'])

            evapzsurf = dataset.createVariable('EvapZsurf', np.float64, ('lat','lon'))
            evapzsurf[:] = np.float64(soil_params['EvapZsurf'])

            evapzmin = dataset.createVariable('EvapZmin', np.float64, ('lat','lon'))
            evapzmin[:] = np.float64(soil_params['EvapZmin'])

            evapzmax = dataset.createVariable('EvapZmax', np.float64, ('lat','lon'))
            evapzmax[:] = np.float64(soil_params['EvapZmax'])

            kex = dataset.createVariable('Kex', np.float64, ('lat','lon'))
            kex[:] = np.float64(soil_params['Kex'])

            fevap = dataset.createVariable('fevap', np.float64, ('lat','lon'))
            fevap[:] = np.float64(soil_params['fevap'])

            fwrelexp = dataset.createVariable('fWrelExp', np.float64, ('lat','lon'))
            fwrelexp[:] = np.float64(soil_params['fWrelExp'])

            fwcc = dataset.createVariable('fwcc', np.float64, ('lat','lon'))
            fwcc[:] = np.float64(soil_params['fwcc'])

            # adjrew = dataset.createVariable('AdjREW', np.int32, ('lat','lon'))
            # adjrew[:] = np.int32(soil_params['AdjREW'])

            rew = dataset.createVariable('REW', np.float64, ('lat','lon'))
            rew[:] = np.float64(soil_params['REW'])

            # adjcn = dataset.createVariable('AdjCN', np.int32, ('lat','lon'))
            # adjcn[:] = np.int32(soil_params['AdjCN'])

            cn = dataset.createVariable('CN', np.float64, ('lat','lon'))
            cn[:] = np.float64(soil_params['CN'])

            zcn = dataset.createVariable('zCN', np.float64, ('lat','lon'))
            zcn[:] = np.float64(soil_params['zCN'])

            zgerm = dataset.createVariable('zGerm', np.float64, ('lat','lon'))
            zgerm[:] = np.float64(soil_params['zGerm'])

            zres = dataset.createVariable('zRes', np.float64, ('lat','lon'))
            zres[:] = np.float64(soil_params['zRes'])

            fshape_cr = dataset.createVariable('fshape_cr', np.float64, ('lat','lon'))
            fshape_cr[:] = np.float64(soil_params['fshape_cr'])

            dataset.close()

            # irrigation schedule
            # ===================

            def perdelta(start, end, delta):
                curr = start
                while curr < end:
                    yield curr
                    curr += delta

            schedule_fn = 'IrrigationSchedule.txt'        
            schedule_path = os.path.join(os.path.join(matlabdir,inputdir), schedule_fn)

            if os.path.isfile(schedule_path):
                d = read_csv(schedule_path, delimiter="\t", header=1)
                col_nms = d.columns.str.replace('%%','')
                col_nms = col_nms.str.strip()
                d.columns = col_nms

                years = config_years
                # years = d['Year'].values
                schedule_years = d['Year'].values
                months = d['Month'].values
                days = d['Day'].values
                depths = d['Irrigation(mm)'].values
                irrigation_dates = []
                for i in range(len(months)):
                    irrigation_dates.append((schedule_years[i],months[i],days[i]))
                # for i in range(len(months)):
                #     irrigation_dates.append((months[i],days[i]))

                year0 = years[0]
                year1 = years[-1] + 1

                dates = []
                for result in perdelta(datetime.datetime(year0, 1, 1), datetime.datetime(year1+1, 1, 1), datetime.timedelta(days=1)):
                    dates.append(result)

                irrigation_amount = []
                for i in range(len(dates)):
                    date = dates[i]
                    if 'Ex7' in test:
                        date_index = (date.month, date.day)
                        new_irrigation_dates = []
                        for tpl in irrigation_dates:
                            new_irrigation_dates.append((tpl[1], tpl[2]))
                    else:
                        new_irrigation_dates = irrigation_dates
                        date_index = (date.year, date.month, date.day)

                    try:
                        idx = new_irrigation_dates.index(date_index)
                        depth = depths[idx]
                    except:
                        depth = 0

                    irrigation_amount.append(depth)

                irrigation_amount = np.array(irrigation_amount)[:,None,None,None] * np.ones((ncrop,nlat,nlon))[None,:,:,:]

            else:
                irrigation_amount = 0.

            dataset = Dataset(os.path.join(pythondir,inputdir,'irrigation_schedule_' + nc_prefix + '.nc'), 'w')
            dataset.description = 'AquaCrop irrigation schedule'
            dataset.history = 'Created ' + time.ctime(time.time())

            # dataset dimensions
            tm = dataset.createDimension('time', None)
            lat = dataset.createDimension('lat', nlat)
            lon = dataset.createDimension('lon', nlon)
            crop = dataset.createDimension('crop', ncrop)

            # dataset variables
            times = dataset.createVariable('time', 'f4', ('time',))
            times.units = "Days since 1901-01-01"
            times.calendar = 'standard'
            times[:] = date2num(dates, units=times.units, calendar=times.calendar)

            latitudes = dataset.createVariable('lat', np.float64, ('lat',))
            latitudes.standard_name = "latitude"
            latitudes.units = "degrees_north"
            latitudes[:] = lat_vals

            longitudes = dataset.createVariable('lon', np.float64, ('lon',))
            longitudes.standard_name = "longitude"
            longitudes.units = "degrees_east"
            longitudes[:] = lon_vals

            crops = dataset.createVariable('crop', np.int32, ('crop',))
            crops[:] = np.arange(0,ncrop,1)

            schd = dataset.createVariable('irrigation_depth', np.float64, ('time','crop','lat','lon'))
            schd.setncattr('standard_name', 'irrigation_depth')
            schd.setncattr('long_name', 'irrigation_depth')
            schd.setncattr('units', '1e-3 m.day-1')
            schd.setncattr('_FillValue', np.float64(1e+20))
            schd.setncattr('missing_value', np.float64(1e+20))
            schd[:] = irrigation_amount

            dataset.close()

            # initial values
            # ==============

            init_cond_fn = os.path.join(os.path.join(matlabdir,inputdir), 'InitialWaterContent.txt')
            with open(init_cond_fn) as f:
                content = f.read().splitlines()

            content = [x for x in content if re.search('^(?!%%).*', x)]
            init_cond_type = content[0]
            init_cond_interp = content[1]
            init_cond_npt = content[2]
            init_cond_data = read_csv(init_cond_fn, delimiter='\s+|\t', header=None, names=['depth_or_layer','value'], skiprows=8, engine='python')

            zLayerSum = np.cumsum(zLayer)
            nr = init_cond_data.shape[0]
            th_vals = np.zeros((nr), dtype=np.float64)

            # get layer index
            # ###############
            if (init_cond_interp == 'Layer'):
                layers = np.array(init_cond_data['depth_or_layer'])
            elif (init_cond_interp == 'Depth'):
                layers = np.zeros((nr))
                for rw in range(nr):
                    if init_cond_interp == 'Depth':
                        dp = init_cond_data['depth_or_layer'][rw]
                        if (dp >= 0) & (dp <= zLayerSum[-1]):
                            lyr = np.argmax(zLayerSum >= dp) + 1
                        elif (dp > zLayerSum[-1]):
                            lyr = nlayer
                        elif (dp < 0):
                            lyr = 1
                        layers[rw] = lyr

            # assign values, depending on type
            # ################################
            for rw in range(nr):
                val = init_cond_data['value'][rw]
                lyr = layers[rw] - 1
                if init_cond_type == 'Prop':
                    if val == 'FC': th_vals[rw] = th_fc[int(lyr)]
                    if val == 'WP': th_vals[rw] = th_wp[int(lyr)]
                    if val == 'SAT': th_vals[rw] = th_s[int(lyr)]
                    # if val == 'FC': init_cond_data.at[rw,'value'] = th_fc[int(lyr)]
                    # if val == 'WP': init_cond_data.at[rw,'value'] = th_wp[int(lyr)]
                    # if val == 'SAT': init_cond_data.at[rw,'value'] = th_s[int(lyr)]

                elif init_cond_type == 'Pct':
                    if (init_cond_interp == 'Depth') | (init_cond_interp == 'Layer'):
                        taw = th_fc[int(lyr)] - th_wp[int(lyr)]
                        th_vals[rw] = th_wp[int(lyr)] + (taw * (np.float64(val) / 100))

                        # init_cond_data.at[rw,'value'] = th_wp[int(lyr)] + (taw * (val / 100))

                elif init_cond_type == 'Num':
                    th_vals[rw] = np.float64(val)

            if (init_cond_interp == 'Layer'):
                thini = np.ones((nlayer,nlat,nlon))
                vals = th_vals.copy()
                # vals = init_cond_data['value'].values
                depths = zLayerMid
                ndepth = nlayer
                if (vals.size == nlayer):
                    for lyr in range(vals.size):
                        thini[lyr,:,:] = vals[lyr]
                else:
                    raise

            elif (init_cond_interp == 'Depth'):
                vals = th_vals.copy()
                # vals = init_cond_data['value']
                depths = init_cond_data['depth_or_layer']
                ndepth = depths.size
                thini = np.ones((ndepth,nlat,nlon))
                for nd in range(ndepth):
                    thini[nd,:,:] = vals[nd]

            dataset = Dataset(os.path.join(pythondir,inputdir,'initial_conditions_' + nc_prefix + '.nc'), 'w')
            dataset.description = 'AquaCrop initial value file'
            dataset.history = 'Created ' + time.ctime(time.time())

            lat = dataset.createDimension('lat', nlat)
            lon = dataset.createDimension('lon', nlon)
            crop = dataset.createDimension('crop', ncrop)
            depth = dataset.createDimension('depth', ndepth)

            latitudes = dataset.createVariable('lat', np.float64, ('lat',))
            latitudes.standard_name = "latitude"
            latitudes.units = "degrees_north"
            latitudes[:] = lat_vals

            longitudes = dataset.createVariable('lon', np.float64, ('lon',))
            longitudes.standard_name = "longitude"
            longitudes.units = "degrees_east"
            longitudes[:] = lon_vals

            crops = dataset.createVariable('crop', np.int32, ('crop',))
            crops[:] = np.arange(0,ncrop,1)

            depth = dataset.createVariable('depth', np.float64, ('depth',))
            depth[:] = np.array(depths)

            soilwatercontent = dataset.createVariable('th', np.float64, ('depth','lat','lon'))
            soilwatercontent[:] = thini

            bundwater = dataset.createVariable('BundWater', np.float64, ('crop','lat','lon'))
            bundwater[:] = np.float64(mgmt_params['BundWater'])

            dataset.close()

            # CO2 data
            # ========

            co2_fn = os.path.join(
                os.path.join(matlabdir,inputdir),
                'MaunaLoaCO2.txt')
            d = read_csv(co2_fn, delimiter="\s+", header=None, skiprows=[0,1], names=['Year','CO2'])

            years = d['Year'].values
            data = d['CO2'].values
            f = interpolate.interp1d(years, data, kind="linear")
            newyears = np.arange(years.min(), years.max() + 1)
            newdata = f(newyears)

            # from scipy import interp
            # max_config_yr = np.max(config_years)
            # if (max_config_yr > years.max()):
            #     index = np.arange(years.min(), max_config_yr + 1)
            #     newdata = interp(index,newyears,newdata)

            dataset = Dataset(os.path.join(pythondir,inputdir,'annual_co2_conc_' + nc_prefix + '.nc'), 'w')
            dataset.description = 'Annual CO2 concentration file'
            dataset.history = 'Created ' + time.ctime(time.time())

            lat = dataset.createDimension('lat', nlat)
            lon = dataset.createDimension('lon', nlon)
            tm = dataset.createDimension('time', None)

            latitudes = dataset.createVariable('lat', np.float64, ('lat',))
            latitudes.standard_name = "latitude"
            latitudes.units = "degrees_north"
            latitudes[:] = lat_vals

            longitudes = dataset.createVariable('lon', np.float64, ('lon',))
            longitudes.standard_name = "longitude"
            longitudes.units = "degrees_east"
            longitudes[:] = lon_vals

            times = dataset.createVariable('time', np.int32, ('time',))
            times.units = "days since 1901-01-01 00:00:00"
            times.calendar = "standard"

            dates = []
            for i in range(len(newyears)):
                dates.append(datetime.datetime(newyears[i],1,1))

            times[:] = date2num(dates, units=times.units, calendar=times.calendar)

            co2 = dataset.createVariable('co2', np.float64, ('time','lat','lon'))
            co2.setncattr('standard_name', 'co2')
            co2.setncattr('long_name', 'CO2 concentration by volume')
            co2.setncattr('units', '1e-6')
            co2.setncattr('_FillValue', np.float64(1e+20))
            co2.setncattr('missing_value', np.float64(1e+20))
            co2[:] = newdata[:,None,None] * np.ones((nlat,nlon))[None,:,:]

            dataset.close()

            # write config file
            # #################

            clock_fn = os.path.join(os.path.join(matlabdir,inputdir), 'Clock.txt')
            clock_info = read_params(clock_fn)
            start_time = clock_info['SimulationStartTime']
            end_time = clock_info['SimulationEndTime']

            import configparser
            import collections
            config = configparser.ConfigParser()
            config.optionxform = str
            config['FILE_PATHS'] = collections.OrderedDict([
                ('PathIn','Input'),
                ('PathOut','Output')
            ])

            config['MODEL_GRID'] = collections.OrderedDict([
                ('cloneMap', '${FILE_PATHS:PathIn}/test.clone.tif'),
                ('landmask', '${FILE_PATHS:PathIn}/test.landmask.tif')
            ])

            config['CLOCK'] = collections.OrderedDict([
                ('startTime', start_time),
                ('endTime', end_time)
            ])

            config['INITIAL_WATER_CONTENT'] = collections.OrderedDict([
                ('initialConditionType', 'FILE'),
                ('initialConditionNC', '${FILE_PATHS:PathIn}' + '/' + 'initial_conditions_' + nc_prefix + '.nc'),
                ('initialConditionInterpMethod', init_cond_interp)
            ])

            config['NETCDF_ATTRIBUTES'] = collections.OrderedDict([
                ('institution', 'University of Exeter, UK'),
                ('title', 'AquaCrop v5.0 output'),
                ('description', 'test version (by Simon Moulds)'),
                ('netcdf_y_orientation_follow_cf_convention', 'True'),
                ('formatNetCDF', 'NETCDF4'),
                ('zlib', 'True')
            ])            

            config['WEATHER'] = collections.OrderedDict([
                ('precipitationNC', '${FILE_PATHS:PathIn}' + '/' + 'prec_' + nc_prefix + '.nc'),
                ('precipitationVarName', 'precipitation'),
                ('precipitationTimeDimName', 'time'),
                ('precipitationOffset', str(0)),
                ('precipitationFactor', str(1)),
                ('meanDailyTemperatureNC', str(None)),
                ('meanDailyTemperatureVarName', str(None)),
                ('meanDailyTemperatureTimeDimName', 'time'),
                ('meanDailyTemperatureOffset', str(0)),
                ('meanDailyTemperatureFactor', str(1)),
                ('minDailyTemperatureNC', '${FILE_PATHS:PathIn}' + '/' + 'temp_' + nc_prefix + '.nc'),
                ('minDailyTemperatureVarName', 'Tmin'),
                ('minDailyTemperatureTimeDimName', 'time'),
                ('minDailyTemperatureOffset', str(0)),
                ('minDailyTemperatureFactor', str(1)),
                ('maxDailyTemperatureNC', '${FILE_PATHS:PathIn}' + '/' + 'temp_' + nc_prefix + '.nc'),
                ('maxDailyTemperatureVarName', 'Tmax'),
                ('maxDailyTemperatureTimeDimName', 'time'),
                ('maxDailyTemperatureOffset', str(0)),
                ('maxDailyTemperatureFactor', str(1)),
                ('refEvapotranspirationNC', '${FILE_PATHS:PathIn}' + '/' + 'eto_' + nc_prefix + '.nc'),
                ('refEvapotranspirationVarName', 'referencePotET'),
                ('refEvapotranspirationTimeDimName', 'time'),
                ('refEvapotranspirationOffset', str(0)),
                ('refEvapotranspirationFactor', str(1))
            ])

            config['CARBON_DIOXIDE'] = collections.OrderedDict([
                ('carbonDioxideNC', '${FILE_PATHS:PathIn}/annual_co2_conc_' + nc_prefix + '.nc')
            ])

            config['WATER_TABLE'] = collections.OrderedDict([
                ('WaterTable', str(0)),
                ('VariableWaterTable', str(0)),
                ('groundwaterVarName', str(None)),
                ('groundwaterInputDir', str(None)),
                ('DailyGroundwaterNC', str(0)),
                ('groundwaterInputFile', str(None))
            ])

            config['LAND_COVER'] = collections.OrderedDict([])

            config['CROP_PARAMETERS'] = collections.OrderedDict([
                ('nCrop', str(ncrop)),
                ('cropParametersNC', '${FILE_PATHS:PathIn}/params_' + nc_prefix + '.nc'),
                ('CalendarType', str(calendar_type)),
                ('SwitchGDD', str(switch_gdd)),
                ('GDDmethod', str(gdd_method)),
                ('outDailyTotal', str('th,Y,Irr,B,IrrCum,IrrNetCum'))
            ])

            config['IRRIGATION_MANAGEMENT'] = collections.OrderedDict([
                ('irrigationManagementNC', '${FILE_PATHS:PathIn}/params_' + nc_prefix + '.nc'),
                ('irrigationScheduleNC', '${FILE_PATHS:PathIn}/irrigation_schedule_' + nc_prefix + '.nc')
            ])

            config['FIELD_MANAGEMENT'] = collections.OrderedDict([
                ('fieldManagementNC', '${FILE_PATHS:PathIn}/params_' + nc_prefix + '.nc')
            ])

            config['SOIL_HYDRAULIC_PARAMETERS'] = collections.OrderedDict([
                ('calculateSoilHydraulicParametersFromSoilTexture', '0'),
                ('soilHydraulicParametersNC', '${FILE_PATHS:PathIn}/params_' + nc_prefix + '.nc'),
                ('saturatedHydraulicConductivityVarName', 'ksat'),
                ('saturatedVolumetricWaterContentVarName', 'th_s'),
                ('fieldCapacityVolumetricWaterContentVarName', 'th_fc'),
                ('wiltingPointVolumetricWaterContentVarName', 'th_wp'),
                ('dzSoilLayer',','.join(['%.5f' % num for num in zLayer])),
                ('dzSoilCompartment',','.join(['%.5f' % num for num in zcomp]))
            ])

            config['SOIL_PARAMETERS'] = collections.OrderedDict([
                ('soilParametersNC', '${FILE_PATHS:PathIn}/params_' + nc_prefix + '.nc'),
                ('adjustReadilyAvailableWater', np.int32(soil_params['AdjREW'])),
                ('adjustCurveNumber', np.int32(soil_params['AdjCN']))
            ])        

            config['REPORTING'] = collections.OrderedDict([
                ('formatNetCDF', 'NETCDF4'),
                ('zlib', 'True')
            ])

            with open(os.path.join(pythondir, configdir, nc_prefix + '_' + str(yr) + '_config.ini'), 'w') as configfile:
                config.write(configfile)

if __name__ == "__main__":
    run()

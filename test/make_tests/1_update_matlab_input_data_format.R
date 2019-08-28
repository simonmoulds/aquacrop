## Author : Simon Moulds
## Date   : July 2018

library(stringr)
library(magrittr)
source("funs.R")

test_configs = c("Ch7_Tunis_Wheat_Ex1a",
                 "Ch7_Tunis_Wheat_Ex1b",
                 "Ch7_Tunis_Wheat_Ex2a",
                 "Ch7_Tunis_Wheat_Ex2b",
                 "Ch7_Tunis_Wheat_Ex3a",
                 "Ch7_Tunis_Wheat_Ex3b",
                 "Ch7_Tunis_Wheat_Ex3c",
                 "Ch7_Tunis_Wheat_Ex3d",
                 ## "Ch7_Tunis_Wheat_Ex4a",
                 ## "Ch7_Tunis_Wheat_Ex4b",
                 "Ch7_Tunis_Wheat_Ex6",
                 "Ch7_Tunis_Wheat_Ex7a",
                 "Ch7_Tunis_Wheat_Ex7b",
                 "Ch7_Tunis_Wheat_Ex7c",
                 "Ch7_Tunis_Wheat_Ex7d",
                 "Ch7_Tunis_Wheat_Ex7e",
                 "Ch8_Hyderabad_Cereal_Ex2a",
                 "Ch8_Hyderabad_Cereal_Ex2b",
                 "Ch8_Hyderabad_Cereal_Ex3a",
                 "Ch8_Hyderabad_Cereal_Ex3b",
                 "Ch8_Hyderabad_Cereal_Ex3c",
                 "Ch8_Hyderabad_Cereal_Ex3d",
                 "Ch8_Hyderabad_Cereal_Ex6",
                 "Ch9_Brussels_Potato_Ex1",
                 ## "Ch9_Brussels_Potato_Ex2",
                 "Ch9_Brussels_Potato_Ex4",
                 "Ch9_Brussels_Potato_Ex5a",
                 "Ch9_Brussels_Potato_Ex5b",
                 "Ch9_Brussels_Potato_Ex5c",
                 "Ch9_Brussels_Potato_Ex6a",
                 "Ch9_Brussels_Potato_Ex6b"
)

## clean directory to store output
testdir = "aquacrop_matlab_test_data"
unlink(testdir, recursive=TRUE)
dir.create(testdir)

## unzip directory containing the various tests
system("unzip -o AquaCropOS_ModelTesting.zip")

## ======================================
## copy some input files for certain exercises
## ======================================

## for Ch7, Ex6, we need to change the simulation so that it
## starts on the planting day

dir = "AquaCropOS_ModelTesting/Chapter7_Tunis_Wheat/Exercise6/Matlab/Input"
y = readLines(file.path(dir, "Clock.txt"))

yy = y
yy[3] = "SimulationStartTime : 2001-12-01-00-00-00"
fileCon = file(file.path(dir, "Clock_Dec01.txt"))
writeLines(yy, fileCon)
close(fileCon)

## for Ch7, Ex7, we need to copy the irrigation schedule file
## ##########################################################

dir = "AquaCropOS_ModelTesting/Chapter7_Tunis_Wheat/Exercise7/Matlab/Input"
x = readLines(file.path(dir, "Irrigation_Schedule.txt"))

## Ex7a (01/12/2001, 01/01/2002, 01/02/2002)
xx = x
xx[3:5] = c("01/12/2001\t30","01/01/2002\t40","01/02/2002\t40")
fileCon = file(file.path(dir, "Irrigation_Schedule_Sch1.txt"))
writeLines(xx, fileCon)
close(fileCon)

## Ex7b (01/12/2001, 01/01/2002, 01/03/2002)
xx = x
xx[3:5] = c("01/12/2001\t30","01/01/2002\t40","01/03/2002\t40")
fileCon = file(file.path(dir, "Irrigation_Schedule_Sch2.txt"))
writeLines(xx, fileCon)
close(fileCon)

## Ex7c (01/12/2001, 15/12/2001, 15/02/2002)
xx = x
xx[3:5] = c("01/12/2001\t30","15/12/2001\t40","15/02/2002\t40")
fileCon = file(file.path(dir, "Irrigation_Schedule_Sch3.txt"))
writeLines(xx, fileCon)
close(fileCon)

## Ex7d (01/12/2001, 01/03/2002, 01/05/2002)
xx = x
xx[3:5] = c("01/12/2001\t30","01/03/2002\t40","01/05/2002\t40")
fileCon = file(file.path(dir, "Irrigation_Schedule_Sch4.txt"))
writeLines(xx, fileCon)
close(fileCon)

## Ex7e (01/12/2001, 15/12/2001, 01/04/2002)
xx = x
xx[3:5] = c("01/12/2001\t30","15/12/2001\t40","01/04/2002\t40")
fileCon = file(file.path(dir, "Irrigation_Schedule_Sch5.txt"))
writeLines(xx, fileCon)
close(fileCon)

## for Ch8, Ex2 we need to copy the planting date file and
## clock file
## ##########

dir = "AquaCropOS_ModelTesting/Chapter8_Hyderabad_Cereal/Exercise2/Matlab/Input"
x = readLines(file.path(dir, "PlantingDates_Hyderabad.txt"))
y = readLines(file.path(dir, "Clock.txt"))

## 1: Ex2a (planting on 15/07)
xx = x
xx[5] = "PlantDate : 15/07"
fileCon = file(file.path(dir, "PlantingDates_Hyderabad_Jul15.txt"))
writeLines(xx, fileCon)
close(fileCon)

yy = y
yy[3] = "SimulationStartTime : 2010-07-15-00-00-00"
fileCon = file(file.path(dir, "Clock_Jul15.txt"))
writeLines(yy, fileCon)
close(fileCon)

## 1: Ex2b (planting on 01/08)
xx = x
xx[5] = "PlantDate : 01/08"
fileCon = file(file.path(dir, "PlantingDates_Hyderabad_Aug01.txt"))
writeLines(xx, fileCon)
close(fileCon)

yy = y
yy[3] = "SimulationStartTime : 2010-08-01-00-00-00"
fileCon = file(file.path(dir, "Clock_Aug01.txt"))
writeLines(yy, fileCon)
close(fileCon)

## ## trying to find whether bunds are causing a problem
## x = readLines(file.path(dir, "IrrigationManagement.txt"))
## x[29] = "zBund : 0.0"
## fileCon = file(file.path(dir, "IrrigationManagement.txt"))
## writeLines(x, fileCon)
## close(fileCon)

## for Ch9, Ex5, concatenate the irrigation schedule
## #################################################

dir = "AquaCropOS_ModelTesting/Chapter9_Brussels_Potato/Exercise5/Matlab/Input"
yrs = 1985:2005

## 1 - RAW36
fs1 = list.files(dir, "IrrigationSchedule_RAW36_[0-9]{4}.txt", full.names=TRUE) %>% sort
x = readLines(fs1[1]) %>% update_irr_schedule_input(yrs[1])
for (i in 2:length(fs1)) {
    x1 = readLines(fs1[i]) %>% update_irr_schedule_input(yrs[i]) %>% `[`(grep("^%%", x=., invert=TRUE))
    x = c(x, x1)
}

fileCon = file(file.path(dir, "IrrigationSchedule_RAW36.txt"))
writeLines(x, fileCon)
close(fileCon)

## 2 - RAW100
fs1 = list.files(dir, "IrrigationSchedule_RAW100_[0-9]{4}.txt", full.names=TRUE) %>% sort
x = readLines(fs1[1]) %>% update_irr_schedule_input(yrs[1])
for (i in 2:length(fs1)) {
    x1 = readLines(fs1[i]) %>% update_irr_schedule_input(yrs[i]) %>% `[`(grep("^%%", x=., invert=TRUE))
    x = c(x, x1)
}

fileCon = file(file.path(dir, "IrrigationSchedule_RAW100.txt"))
writeLines(x, fileCon)
close(fileCon)

## 3 - RAW150
fs1 = list.files(dir, "IrrigationSchedule_RAW150_[0-9]{4}.txt", full.names=TRUE) %>% sort
x = readLines(fs1[1]) %>% update_irr_schedule_input(yrs[1])
for (i in 2:length(fs1)) {
    x1 = readLines(fs1[i]) %>% update_irr_schedule_input(yrs[i]) %>% `[`(grep("^%%", x=., invert=TRUE))
    x = c(x, x1)
}

fileCon = file(file.path(dir, "IrrigationSchedule_RAW150.txt"))
writeLines(x, fileCon)
close(fileCon)

## for Ch9, Ex6, extend co2 data
## #############################

dir = "AquaCropOS_ModelTesting/Chapter9_Brussels_Potato/Exercise6/Matlab/Input"
fn = file.path(dir, "MaunaLoaCO2.txt")
x = readLines(fn) # %>% update_co2_input
header = x[1:2]
dat = x[-(1:2)] %>% gsub("^\\s+|\\s+$", "", .) %>% strsplit("\\s+")
year=as.numeric(sapply(dat, FUN=function(x) x[1]))
data=as.numeric(sapply(dat, FUN=function(x) x[2]))

newyear = seq(min(year), 2070)
newdata = rep(NA, length(newyear))
newdata[1:length(data)] = data

library(tidyr)
library(dplyr)
df = data.frame(year=newyear, data=newdata) %>% zoo::na.approx(rule=2) %>% as.data.frame %>% unite(col=dat, sep="  ") 
xx = c(header, df[["dat"]])
                
fileCon = file(file.path(dir, "MaunaLoaCO2.txt"))
writeLines(xx, fileCon)
close(fileCon)

for (test_config in test_configs) {

    print(test_config)
    
    vals = config::get(config=test_config)
    attach(vals)

    dir.create(file.path(testdir, exdirnm))
    inpdirnm = "Input"
    outdirnm = "Output"               
    ## dir.create(file.path(testdir, exdirnm, inpdirnm))
    ## dir.create(file.path(testdir, exdirnm, outdirnm))
    for (year in years) {
        dir.create(file.path(testdir, exdirnm, year))
        inpdir = file.path(testdir, exdirnm, year, inpdirnm)
        outdir = file.path(testdir, exdirnm, year, outdirnm)
        dir.create(inpdir)
        dir.create(outdir)
        ## ## inpdir = file.path(testdir, paste0(inpdirnm, "_", year))
        ## if (!dir.exists(inpdir)) {
        ##     dir.create(inpdir)
        ## }

        ## ## use the same output directory
        ## outdir = file.path(testdir, exdirnm, year, outdirnm)
        ## ## outdir = file.path(testdir, outdirnm)
        ## if (!dir.exists(outdir)) {
        ##     dir.create(outdir)
        ## }

        ## Clock
        ## #####
        fn = file.path(wd, "Input", old_clock_fn)
        x = readLines(fn) %>% update_clock_input
        ix0 = grep("^SimulationStartTime", x)
        startdate =
            x %>% `[`(ix0) %>%
            strsplit(":") %>% `[[`(1) %>% `[`(2) %>%
            gsub("^(\\s+)","",.) 
        ix1 = grep("^SimulationEndTime", x)
        enddate =
            x %>% `[`(ix1) %>%
            strsplit(":") %>% `[[`(1) %>% `[`(2) %>%
            gsub("^(\\s+)","",.) 
        startyear = startdate %>% substr(1,4) %>% as.numeric
        endyear = enddate %>% substr(1,4) %>% as.numeric
        newstartyear = year
        newendyear = year + (endyear - startyear)
        newstartdate = gsub("^[0-9]{4}",newstartyear,startdate) 
        newenddate = gsub("^[0-9]{4}",newendyear,enddate) 
        x[ix0] = paste0("SimulationStartTime : ", newstartdate)
        x[ix1] = paste0("SimulationEndTime : ", newenddate)
        fileCon = file(file.path(inpdir, new_clock_fn))
        writeLines(x, fileCon)
        close(fileCon)

        ## Initial water content
        ## #####################
        fn = file.path(wd, "Input", old_initial_wc_fn)
        x = readLines(fn) %>% update_initial_wc_input
        fileCon = file(file.path(inpdir, new_initial_wc_fn))
        writeLines(x, fileCon)
        close(fileCon)

        ## Irrigation schedule
        ## ###################        
        fn = try(file.path(wd, "Input", old_irri_schedule_fn), silent=TRUE)
        if (!"try-error" %in% class(fn)) {
            x = readLines(fn) %>% update_irr_schedule_input(year)
            fileCon = file(file.path(inpdir, new_irri_schedule_fn))
            writeLines(x, fileCon)
            close(fileCon)
        }
                
        ## Irrigation management
        ## #####################
        fn = file.path(wd, "Input", old_irri_mgmt_fn)
        x = readLines(fn) %>% update_irr_mgmt_input
        fileCon = file(file.path(inpdir, new_irri_mgmt_fn))
        writeLines(x, fileCon)
        close(fileCon)
        
        ## MaunaLoaCO2
        ## ###########
        fn = file.path(wd, "Input", old_co2_fn)
        x = readLines(fn) %>% update_co2_input
        fileCon = file(file.path(inpdir, new_co2_fn))
        writeLines(x, fileCon)
        close(fileCon)

        ## Field management
        ## ################
        fn1 = file.path(wd, "Input", old_soil_param_fn)
        fn2 = file.path(wd, "Input", old_irri_mgmt_fn)
        x = c(readLines(fn1),readLines(fn2)) %>% update_field_mgmt_input
        fileCon = file(file.path(inpdir, new_field_mgmt_fn))
        writeLines(x, fileCon)
        close(fileCon)

        ## Weather data
        ## ############
        fn = file.path(wd, "Input", old_weather_fn)
        x = readLines(fn) %>% update_weather_input

        fn = file.path(inpdir, new_weather_fn)
        fileCon = file(fn)
        writeLines(x, fileCon)
        close(fileCon)

        ## Crop parameters
        ## ###############

        ## 1 - crop mix file
        ncrop = 1
        planting_calendar = "N"
        crop_rotation_filename = "CropRotation.txt"
        crop_nm = "Wheat"
        crop_param_fn = new_crop_param_fn
        irri_mgmt_fn = new_irri_mgmt_fn

        x = update_crop_mix_input(ncrop, planting_calendar, crop_rotation_filename, crop_nm, crop_param_fn, irri_mgmt_fn)
        fileCon = file(file.path(inpdir, new_crop_mix_fn))
        writeLines(x, fileCon)
        close(fileCon)

        ## 2 - crop parameter file
        fn1 = file.path(wd, "Input", old_crop_param_fn)
        fn2 = file.path(wd, "Input", old_planting_date_fn)
        x = c(readLines(fn1), readLines(fn2)) %>% update_crop_param_input
        fileCon = file(file.path(inpdir, new_crop_param_fn))
        writeLines(x, fileCon)
        close(fileCon)

        ## soil parameters
        ## ###############
        fn = file.path(wd, "Input", old_soil_param_fn)
        x = readLines(fn) %>% update_soil_param_input
        fileCon = file(file.path(inpdir, new_soil_param_fn))
        writeLines(x, fileCon)
        close(fileCon)

        ## Soil profile
        ## ############
        fn = file.path(wd, "Input", old_soil_profile_fn)
        x = readLines(fn) %>% update_soil_profile_input
        fileCon = file(file.path(inpdir, new_soil_profile_fn))
        writeLines(x, fileCon)
        close(fileCon)

        ## Soil texture
        ## ############

        ## NB soil texture information is not used

        ## Soil hydrology
        ## ##############

        ## first, get profile thicknesses from SoilProfile.txt
        fn = file.path(wd, "Input", old_soil_profile_fn)
        layer_thickness = readLines(fn) %>% compute_layer_thickness

        ## now read soil hydrology data
        fn = file.path(wd, "Input", old_soil_hydrology_fn)
        x = readLines(fn) %>% update_soil_hydrology_input(layer_thickness)

        fn = file.path(inpdir, new_soil_hydrology_fn)
        fileCon = file(fn)
        writeLines(x, fileCon)
        close(fileCon)

        ## Water table
        ## ###########

        x = c("%% --------- Groundwater table for AquaCropOS ---------- %%",
              "%% Water table present ('Y' or 'N') %%",
              "N",
              "%% Water table method ('Constant' depth; 'Variable' depth) %%",
              "Constant",
              "%% Date (dd/mm/yyyy)   Depth (m) %%",
              "01/09/2000  2")

        fileCon = file(file.path(inpdir, new_water_table_fn))
        writeLines(x, fileCon)
        close(fileCon)

        ## File setup
        ## ##########
        x = c("%% ----------- Input filenames for AquaCropOS ----------- %%",
              "%% Clock Filename %%",
              new_clock_fn,
              "%% Weather Data Filename %%",
              new_weather_fn,
              "%% C02 Concentration Time-Series Filename %%",
              new_co2_fn,
              "%% Crop Parameters Filename %%",
              new_crop_mix_fn,
              "%% Soil Parameters Filename %%",
              new_soil_param_fn,
              "%% Field Management Filename %%",
              new_field_mgmt_fn,
              "%% Initial Water Content Filename %%",
              new_initial_wc_fn,
              "%% Groundwater Table Filename %%",
              new_water_table_fn,
              "%% Output Filename %%",
              paste0(new_output_fn, "_", year),
              "%% Write daily outputs ('Y' or 'N') %%",
              new_write_daily_output)

        fileCon = file(file.path(inpdir, new_file_setup_fn))
        writeLines(x, fileCon)
        close(fileCon)

        ## original output
        ## ###############
        fs = list.files(file.path(wd, "Output", "Raw"), pattern=paste0("^.*", year, ".*", "(CropGrowth|FinalOutput|WaterContents|WaterFluxes).txt$"))
        for (f in fs) {
            inpath = file.path(wd, "Output", "Raw", f)
            outpath = file.path(outdir, sub(".(txt$)", "_orig.\\1", f))
            file.copy(inpath, outpath, overwrite=TRUE)
        }
    }
    detach(vals)
}

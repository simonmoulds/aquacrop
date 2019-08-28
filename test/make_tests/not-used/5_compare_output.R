## Author : Simon Moulds
## Date   : July 2018

library(tidyverse)
library(magrittr)
library(ncdf4)
library(gridExtra)

aosdir = "/home/simon/projects/AquaCrop/AquaCrop_tests/aquacrop_matlab_test_data/"
aqpydir = "/home/simon/projects/AquaCrop/AquaCrop_tests/aquacrop_python_test_data/"

## a couple of helper functions to retrieve output data
get_aos_output = function(aosdir, outputdir, fn, var, ...) {
    fn = file.path(aosdir, outputdir, fn)
    if (file.exists(fn)) {
        x = read.table(fn, header=TRUE, fill=TRUE, sep="\t")
        out = x[[var]]
    } else {
        out = NA
    }    
    out
}

get_aqpy_output = function(aqpydir, outputdir, fn, ...) {
    fn = file.path(aqpydir, outputdir, "netcdf", fn)
    nc = nc_open(fn)
    nm = attributes(nc$var)$names[1]
    var = ncvar_get(nc, nm)[1,1,]
    nc_close(nc)
    var
}

## ======================================
## Test 1: Ch7, Tunis Wheat, Ex1
## ======================================

## Wheat production in Tunis, Tunisia

## Test 1a: Local soil
## ###################

## yield
years = 1979:2001
yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)
for (year in years) {
    aos_v0_fn = file.path(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex1a",
        paste0("Exercise_7_1_LocalSoil_", year, "_FinalOutput_orig.txt"))
    aos_v0 = read.table(aos_v0_fn, header=TRUE)
    aos_v0_yield = aos_v0$Yield[1]             

    aos_v5_fn = file.path(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex1a",
        paste0("Ch7_Tunis_Wheat_Ex1a_", year, "_FinalOutput.txt"))
    aos_v5 = read.table(aos_v5_fn, header=TRUE)
    aos_v5_yield = aos_v5$Yield[1]

    aqpy_fn = file.path(
        aqpydir,
        paste0("Output_Ch7_Tunis_Wheat_Ex1a_", year),
        "netcdf",
        "Y_dailyTot_output.nc")

    nc = nc_open(aqpy_fn)
    nm = attributes(nc$var)$names[1]
    var = ncvar_get(nc, nm)[1,1,]
    nc_close(nc)
    aqpy_yield = round(max(var), digits=2)

    row = which(yield_comparison$year %in% year)
    yield_comparison$aos_v0[row] = aos_v0_yield
    yield_comparison$aos_v5[row] = aos_v5_yield
    yield_comparison$aqpy[row] = aqpy_yield
}

x = yield_comparison$aos_v5
y = yield_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p1 =
    data.frame(time=years, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Local soil") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    scale_y_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p1

## Test 1b: Sandy Loam
## ###################

## yield
years = 1979:2001
yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)
for (year in years) {
    aos_v0_fn = file.path(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex1b",
        paste0("Exercise_7_1_SandyLoam_", year, "_FinalOutput_orig.txt"))
    aos_v0 = read.table(aos_v0_fn, header=TRUE)
    aos_v0_yield = aos_v0$Yield[1]             

    aos_v5_fn = file.path(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex1b",
        paste0("Ch7_Tunis_Wheat_Ex1b_", year, "_FinalOutput.txt"))
    if (file.exists(aos_v5_fn)) {
        aos_v5 = read.table(aos_v5_fn, header=TRUE)
        aos_v5_yield = aos_v5$Yield[1]
    } else {
        aos_v5_yield = NA
    }    

    aqpy_fn = file.path(
        aqpydir,
        paste0("Output_Ch7_Tunis_Wheat_Ex1b_", year),
        "netcdf",
        "Y_dailyTot_output.nc")

    nc = nc_open(aqpy_fn)
    nm = attributes(nc$var)$names[1]
    var = ncvar_get(nc, nm)[1,1,]
    nc_close(nc)
    aqpy_yield = round(max(var), digits=2)

    row = which(yield_comparison$year %in% year)
    yield_comparison$aos_v0[row] = aos_v0_yield
    yield_comparison$aos_v5[row] = aos_v5_yield
    yield_comparison$aqpy[row] = aqpy_yield
}

x = yield_comparison$aos_v5
y = yield_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p2 =
    data.frame(time=years, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Sandy loam soil") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    scale_y_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p2

grid.arrange(p2,p1,ncol=2)

## ======================================
## Test 2: Ch7, Tunis Wheat, Ex2
## ======================================

## Wheat production in Tunis, Tunisia

## Test 2a: Default crop cycle
## ###########################

years = 1979:2001
yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)
for (year in years) {
    aos_v0_fn = file.path(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex2a",
        paste0("Exercise_7_2_LongGDD_", year, "_FinalOutput_orig.txt"))
    aos_v0 = read.table(aos_v0_fn, header=TRUE)
    aos_v0_yield = aos_v0$Yield[1]             

    aos_v5_fn = file.path(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex2a",
        paste0("Ch7_Tunis_Wheat_Ex2a_", year, "_FinalOutput.txt"))
    aos_v5 = read.table(aos_v5_fn, header=TRUE)
    aos_v5_yield = aos_v5$Yield[1]

    aqpy_fn = file.path(
        aqpydir,
        paste0("Output_Ch7_Tunis_Wheat_Ex2a_", year),
        "netcdf",
        "Y_dailyTot_output.nc")

    nc = nc_open(aqpy_fn)
    nm = attributes(nc$var)$names[1]
    var = ncvar_get(nc, nm)[1,1,]
    nc_close(nc)
    aqpy_yield = round(max(var), digits=2)

    row = which(yield_comparison$year %in% year)
    yield_comparison$aos_v0[row] = aos_v0_yield
    yield_comparison$aos_v5[row] = aos_v5_yield
    yield_comparison$aqpy[row] = aqpy_yield
}

x = yield_comparison$aos_v5
y = yield_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p1 =
    data.frame(time=years, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Default crop cycle") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    scale_y_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p1

## Test 2b: Long crop cycle
## ########################

years = 1979:2001
yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)
for (year in years) {
    aos_v0_fn = file.path(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex2b",
        paste0("Exercise_7_2_LongGDD_", year, "_FinalOutput_orig.txt"))
    aos_v0 = read.table(aos_v0_fn, header=TRUE)
    aos_v0_yield = aos_v0$Yield[1]             

    aos_v5_fn = file.path(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex2b",
        paste0("Ch7_Tunis_Wheat_Ex2b_", year, "_FinalOutput.txt"))
    aos_v5 = read.table(aos_v5_fn, header=TRUE)
    aos_v5_yield = aos_v5$Yield[1]

    aqpy_fn = file.path(
        aqpydir,
        paste0("Output_Ch7_Tunis_Wheat_Ex2b_", year),
        "netcdf",
        "Y_dailyTot_output.nc")

    nc = nc_open(aqpy_fn)
    nm = attributes(nc$var)$names[1]
    var = ncvar_get(nc, nm)[1,1,]
    nc_close(nc)
    aqpy_yield = round(max(var), digits=2)

    row = which(yield_comparison$year %in% year)
    yield_comparison$aos_v0[row] = aos_v0_yield
    yield_comparison$aos_v5[row] = aos_v5_yield
    yield_comparison$aqpy[row] = aqpy_yield
}

x = yield_comparison$aos_v5
y = yield_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p2 =
    data.frame(time=years, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Long crop cycle") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    scale_y_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p2

grid.arrange(p1,p2,ncol=2)

## ======================================
## Test 3: Ch7, Tunis Wheat, Ex3
## ======================================

yield_comparison = data.frame(scenario=1:4,aos_v0=NA,aos_v5=NA,aqpy=NA)
biomass_comparison = data.frame(scenario=1:4,aos_v0=NA,aos_v5=NA,aqpy=NA)

## WetDry
## ######

aos_orig_suffix = c("WetDry","FC","TAW30","TAW75")
suffix = c("Ex3a","Ex3b","Ex3c","Ex3d")
for (i in 1:4) {
    
    ## v0
    aos_v0_yield = get_aos_output(
        aosdir,
        paste0("Output_Ch7_Tunis_Wheat_", suffix[i]),
        paste0("Exercise_7_3_", aos_orig_suffix[i], "_1988_FinalOutput_orig.txt"),
        "Yield")

    aos_v0_biomass =
        get_aos_output(
            aosdir,
            paste0("Output_Ch7_Tunis_Wheat_", suffix[i]),
            paste0("Exercise_7_3_", aos_orig_suffix[i], "_1988_CropGrowth_orig.txt"),
            "Biomass") %>%
        max

    ## v5
    aos_v5_yield = get_aos_output(
        aosdir,
        paste0("Output_Ch7_Tunis_Wheat_", suffix[i]),
        paste0("Ch7_Tunis_Wheat_", suffix[i], "_1988_FinalOutput.txt"),
        "Yield")

    ## v5: biomass
    aos_v5_biomass = get_aos_output(
        aosdir,
        paste0("Output_Ch7_Tunis_Wheat_", suffix[i]),
        paste0("Ch7_Tunis_Wheat_", suffix[i], "_1988_CropGrowth.txt"),
        "Bio") %>%
        max

    aqpy_yield =
        get_aqpy_output(
            aqpydir,
            paste0("Output_Ch7_Tunis_Wheat_", suffix[i], "_1988"),
            "Y_dailyTot_output.nc") %>%
        max %>% round(digits=2)

    aqpy_biomass =
        get_aqpy_output(
            aqpydir,
            paste0("Output_Ch7_Tunis_Wheat_", suffix[i], "_1988"),
            "B_dailyTot_output.nc") %>%
        max %>% round(digits=2)

    yield_comparison$aos_v0[i] = aos_v0_yield
    yield_comparison$aos_v5[i] = aos_v5_yield
    yield_comparison$aqpy[i] = aqpy_yield

    biomass_comparison$aos_v0[i] = aos_v0_biomass
    biomass_comparison$aos_v5[i] = aos_v5_biomass
    biomass_comparison$aqpy[i] = aqpy_biomass
}

## plot yield
scenarios = yield_comparison$scenario
x = yield_comparison$aos_v5
y = yield_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p1 =
    data.frame(time=scenarios, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Yield") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    scale_y_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p1

## plot biomass
scenarios = biomass_comparison$scenario
x = biomass_comparison$aos_v5 / 100     # ???
y = biomass_comparison$aqpy / 100       # ???
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p2 =
    data.frame(time=scenarios, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Biomass") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,20,2), limits=c(0,20)) +
    scale_y_continuous(breaks=seq(0,20,2), limits=c(0,20)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p2

grid.arrange(p2,p1,ncol=2)

## ======================================
## Test 4: Ch7, Tunis Wheat, Ex4
## ======================================

## TODO: add capacity to endogenously determine crop planting
##       date based on rainfall thresholds

## ======================================
## Test 5: Ch7, Tunis Wheat, Ex6
## ======================================

years = 1979:2001
yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)
netirr_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)

for (year in years) {

    ## yield
    aos_v0_yield = get_aos_output(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex6",
        paste0("Exercise_7_6_NetIrr_", year, "_FinalOutput_orig.txt"),
        "Yield")

    aos_v5_yield = get_aos_output(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex6",
        paste0("Ch7_Tunis_Wheat_Ex6_", year, "_FinalOutput.txt"),
        "Yield")

    aqpy_yield = 
        get_aqpy_output(
            aqpydir,
            paste0("Output_Ch7_Tunis_Wheat_Ex6_", year),
            "Y_dailyTot_output.nc") %>%
        max %>% round(digits=2)

    ## net irr
    aos_v0_netirr = get_aos_output(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex6",
        paste0("Exercise_7_6_NetIrr_", year, "_WaterFluxes_orig.txt"),
        "NetIrrigation") %>% sum
        
    aos_v5_netirr = get_aos_output(
        aosdir,
        "Output_Ch7_Tunis_Wheat_Ex6",
        paste0("Ch7_Tunis_Wheat_Ex6_", year, "_WaterFluxes.txt"),
        "Irr") %>% sum

    aqpy_netirr =
        get_aqpy_output(
            aqpydir,
            paste0("Output_Ch7_Tunis_Wheat_Ex6_", year),
            "IrrNetCum_dailyTot_output.nc") %>%
        max %>% round(digits=2)
        
    row = which(yield_comparison$year %in% year)
    yield_comparison$aos_v0[row] = aos_v0_yield
    yield_comparison$aos_v5[row] = aos_v5_yield
    yield_comparison$aqpy[row] = aqpy_yield

    netirr_comparison$aos_v0[row] = aos_v0_netirr
    netirr_comparison$aos_v5[row] = aos_v5_netirr
    netirr_comparison$aqpy[row] = aqpy_netirr
}

## yield
x = yield_comparison$aos_v5
y = yield_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p1 =
    data.frame(time=years, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Crop yield") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    scale_y_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p1

## net irrigation
x = netirr_comparison$aos_v5
y = netirr_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p2 =
    data.frame(time=years, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Net irrigation") +
    xlab('AquaCrop-OS (mm)') +
    ylab('AquaCrop-Py (mm)') +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,500,100), limits=c(0,500)) +
    scale_y_continuous(breaks=seq(0,500,100), limits=c(0,500)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p2

grid.arrange(p1,p2,ncol=2)

## ======================================
## Test 6: Ch7, Tunis Wheat, Ex7
## ======================================

years = 1979:2001
aos_orig_suffix = c("Sch1","Sch2","Sch3","Sch4","Sch5")
suffix = c("Ex7a","Ex7b","Ex7c","Ex7d","Ex7e")
plot_titles = paste0("Schedule ", 1:5)
plots=list()

for (i in 1:length(suffix)) {
    
    yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)

    for (j in 1:length(years)) {
        
        ## yield
        aos_v0_yield = get_aos_output(
            aosdir,
            paste0("Output_Ch7_Tunis_Wheat_", suffix[i]),
            paste0("Exercise_7_7_", aos_orig_suffix[i], "_", years[j], "_FinalOutput_orig.txt"),
            "Yield")

        aos_v5_yield = get_aos_output(
            aosdir,
            paste0("Output_Ch7_Tunis_Wheat_", suffix[i]),
            paste0("Ch7_Tunis_Wheat_", suffix[i], "_", years[j], "_FinalOutput.txt"),
            "Yield")

        aqpy_yield = 
            get_aqpy_output(
                aqpydir,
                paste0("Output_Ch7_Tunis_Wheat_", suffix[i], "_", years[j]),
                "Y_dailyTot_output.nc") %>%
            max %>% round(digits=2)

        row = which(yield_comparison$year %in% years[j])
        yield_comparison$aos_v0[row] = aos_v0_yield
        yield_comparison$aos_v5[row] = aos_v5_yield
        yield_comparison$aqpy[row] = aqpy_yield
    }

    ## yield
    x = yield_comparison$aos_v5
    y = yield_comparison$aqpy
    rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
    rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
    p1 =
        data.frame(time=years, x=x, y=y) %>%
        ggplot(aes(x,y)) +
        geom_point() +
        coord_fixed() +
        ggtitle(plot_titles[i]) +
        ## ggtitle("Crop yield") +
        xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
        ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
        geom_abline(slope=1, intercept=c(0,0)) +
        scale_x_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
        scale_y_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
        ## annotate("text",x=0.25,y=9.75,label="hello")
        theme(plot.title=element_text(hjust=0.5))
    plots[[i]] = p1
}

grid.arrange(plots[[1]],plots[[2]],plots[[3]],plots[[4]],plots[[5]],ncol=2)

## ======================================
## Test 7: Ch8, Hyderabad Cereal, Ex2
## ======================================

years = 2000:2010
aos_orig_suffix = c("Jul15","Aug1")
suffix = c("Ex2a","Ex2b")
plot_titles = c("Planting on July 15","Planting on August 1")
plots=list()

for (i in 1:length(suffix)) {
    
    yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)

    for (j in 1:length(years)) {
        
        ## yield
        aos_v0_yield = get_aos_output(
            aosdir,
            paste0("Output_Ch8_Hyderabad_Cereal_", suffix[i]),
            paste0("Exercise_8_2_", aos_orig_suffix[i], "_", years[j], "_FinalOutput_orig.txt"),
            "Yield")

        aos_v5_yield = get_aos_output(
            aosdir,
            paste0("Output_Ch8_Hyderabad_Cereal_", suffix[i]),
            paste0("Ch8_Hyderabad_Cereal_", suffix[i], "_", years[j], "_FinalOutput.txt"),
            "Yield")

        aqpy_yield = 
            get_aqpy_output(
                aqpydir,
                paste0("Output_Ch8_Hyderabad_Cereal_", suffix[i], "_", years[j]),
                "Y_dailyTot_output.nc") %>%
            max %>% round(digits=2)

        row = which(yield_comparison$year %in% years[j])
        yield_comparison$aos_v0[row] = aos_v0_yield
        yield_comparison$aos_v5[row] = aos_v5_yield
        yield_comparison$aqpy[row] = aqpy_yield
    }

    ## yield
    x = yield_comparison$aos_v5
    y = yield_comparison$aqpy
    rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
    rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
    p1 =
        data.frame(time=years, x=x, y=y) %>%
        ggplot(aes(x,y)) +
        geom_point() +
        coord_fixed() +
        ggtitle(plot_titles[i]) +
        ## ggtitle("Crop yield") +
        xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
        ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
        geom_abline(slope=1, intercept=c(0,0)) +
        scale_x_continuous(breaks=seq(0,8,2), limits=c(0,8)) +
        scale_y_continuous(breaks=seq(0,8,2), limits=c(0,8)) +
        ## annotate("text",x=0.25,y=9.75,label="hello")
        theme(plot.title=element_text(hjust=0.5))
    plots[[i]] = p1
}

grid.arrange(plots[[1]],plots[[2]],ncol=2)

## ======================================
## Test 8: Ch8, Hyderabad Cereal, Ex3
## ======================================

yield_comparison = data.frame(scenario=1:4,aos_v0=NA,aos_v5=NA,aqpy=NA)
biomass_comparison = data.frame(scenario=1:4,aos_v0=NA,aos_v5=NA,aqpy=NA)

aos_orig_suffix = c("WetTop","FC","TAW30","TAW75")
suffix = c("Ex3a","Ex3b","Ex3c","Ex3d")

for (i in 1:4) {
    
    ## v0
    aos_v0_yield = get_aos_output(
        aosdir,
        paste0("Output_Ch8_Hyderabad_Cereal_", suffix[i]),
        paste0("Exercise_8_3_", aos_orig_suffix[i], "_2002_FinalOutput_orig.txt"),
        "Yield")

    aos_v0_biomass =
        get_aos_output(
            aosdir,
            paste0("Output_Ch8_Hyderabad_Cereal_", suffix[i]),
            paste0("Exercise_8_3_", aos_orig_suffix[i], "_2002_CropGrowth_orig.txt"),
            "Biomass") %>%
        max

    ## v5
    aos_v5_yield = get_aos_output(
        aosdir,
        paste0("Output_Ch8_Hyderabad_Cereal_", suffix[i]),
        paste0("Ch8_Hyderabad_Cereal_", suffix[i], "_2002_FinalOutput.txt"),
        "Yield")

    ## v5: biomass
    aos_v5_biomass = get_aos_output(
        aosdir,
        paste0("Output_Ch8_Hyderabad_Cereal_", suffix[i]),
        paste0("Ch8_Hyderabad_Cereal_", suffix[i], "_2002_CropGrowth.txt"),
        "Bio") %>%
        max

    aqpy_yield =
        get_aqpy_output(
            aqpydir,
            paste0("Output_Ch8_Hyderabad_Cereal_", suffix[i], "_2002"),
            "Y_dailyTot_output.nc") %>%
        max %>% round(digits=2)

    aqpy_biomass =
        get_aqpy_output(
            aqpydir,
            paste0("Output_Ch8_Hyderabad_Cereal_", suffix[i], "_2002"),
            "B_dailyTot_output.nc") %>%
        max %>% round(digits=2)

    yield_comparison$aos_v0[i] = aos_v0_yield
    yield_comparison$aos_v5[i] = aos_v5_yield
    yield_comparison$aqpy[i] = aqpy_yield

    biomass_comparison$aos_v0[i] = aos_v0_biomass
    biomass_comparison$aos_v5[i] = aos_v5_biomass
    biomass_comparison$aqpy[i] = aqpy_biomass
}

## plot yield
scenarios = yield_comparison$scenario
x = yield_comparison$aos_v5
y = yield_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p1 =
    data.frame(time=scenarios, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Yield") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,8,2), limits=c(0,8)) +
    scale_y_continuous(breaks=seq(0,8,2), limits=c(0,8)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p1

## plot biomass
scenarios = biomass_comparison$scenario
x = biomass_comparison$aos_v5 / 100     # ???
y = biomass_comparison$aqpy / 100       # ???
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p2 =
    data.frame(time=scenarios, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Biomass") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,15,2), limits=c(0,15)) +
    scale_y_continuous(breaks=seq(0,15,2), limits=c(0,15)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p2

grid.arrange(p2,p1,ncol=2)

## ======================================
## Test 9: Ch8, Hyderabad Cereal, Ex6
## ======================================

years = 2000:2009
yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)
netirr_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)

for (year in years) {

    ## yield
    aos_v0_yield = get_aos_output(
        aosdir,
        "Output_Ch8_Hyderabad_Cereal_Ex6",
        paste0("Exercise_8_6_NetIrr_", year, "_FinalOutput_orig.txt"),
        "Yield")

    aos_v5_yield = get_aos_output(
        aosdir,
        "Output_Ch8_Hyderabad_Cereal_Ex6",
        paste0("Ch8_Hyderabad_Cereal_Ex6_", year, "_FinalOutput.txt"),
        "Yield")

    aqpy_yield = 
        get_aqpy_output(
            aqpydir,
            paste0("Output_Ch8_Hyderabad_Cereal_Ex6_", year),
            "Y_dailyTot_output.nc") %>%
        max %>% round(digits=2)

    ## net irr
    aos_v0_netirr = get_aos_output(
        aosdir,
        "Output_Ch8_Hyderabad_Cereal_Ex6",
        paste0("Exercise_8_6_NetIrr_", year, "_WaterFluxes_orig.txt"),
        "NetIrrigation") %>% sum
        
    aos_v5_netirr = get_aos_output(
        aosdir,
        "Output_Ch8_Hyderabad_Cereal_Ex6",
        paste0("Ch8_Hyderabad_Cereal_Ex6_", year, "_WaterFluxes.txt"),
        "Irr") %>% sum

    aqpy_netirr =
        get_aqpy_output(
            aqpydir,
            paste0("Output_Ch8_Hyderabad_Cereal_Ex6_", year),
            "IrrNetCum_dailyTot_output.nc") %>%
        max %>% round(digits=2)
        
    row = which(yield_comparison$year %in% year)
    yield_comparison$aos_v0[row] = aos_v0_yield
    yield_comparison$aos_v5[row] = aos_v5_yield
    yield_comparison$aqpy[row] = aqpy_yield

    netirr_comparison$aos_v0[row] = aos_v0_netirr
    netirr_comparison$aos_v5[row] = aos_v5_netirr
    netirr_comparison$aqpy[row] = aqpy_netirr
}

## yield
x = yield_comparison$aos_v5
y = yield_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p1 =
    data.frame(time=years, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Crop yield") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    scale_y_continuous(breaks=seq(0,10,2), limits=c(0,10)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p1

## net irrigation
x = netirr_comparison$aos_v5
y = netirr_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p2 =
    data.frame(time=years, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Net irrigation") +
    xlab('AquaCrop-OS (mm)') +
    ylab('AquaCrop-Py (mm)') +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,500,100), limits=c(0,500)) +
    scale_y_continuous(breaks=seq(0,500,100), limits=c(0,500)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p2

grid.arrange(p1,p2,ncol=2)

## ======================================
## Test 10: Ch9, Brussels Potato, Ex1
## ======================================

years = 1985:2005
yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)

for (j in 1:length(years)) {

    ## yield
    aos_v0_yield = get_aos_output(
        aosdir,
        "Output_Ch9_Brussels_Potato_Ex1",
        paste0("Exercise_9_1_Local_", years[j], "_FinalOutput_orig.txt"),
        "Yield")

    aos_v5_yield = get_aos_output(
        aosdir,
        "Output_Ch9_Brussels_Potato_Ex1",
        paste0("Ch9_Brussels_Potato_Ex1_", years[j], "_FinalOutput.txt"),
        "Yield")

    aqpy_yield = 
        get_aqpy_output(
            aqpydir,
            paste0("Output_Ch9_Brussels_Potato_Ex1_", years[j]),
            "Y_dailyTot_output.nc") %>%
        max %>% round(digits=2)

    row = which(yield_comparison$year %in% years[j])
    yield_comparison$aos_v0[row] = aos_v0_yield
    yield_comparison$aos_v5[row] = aos_v5_yield
    yield_comparison$aqpy[row] = aqpy_yield
}

## yield
x = yield_comparison$aos_v5
y = yield_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p1 =
    data.frame(time=years, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("") +
    ## ggtitle("Crop yield") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,12,2), limits=c(0,12)) +
    scale_y_continuous(breaks=seq(0,12,2), limits=c(0,12)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p1

## ======================================
## Test 11: Ch9, Brussels Potato, Ex2
## ======================================

## TODO: planting dates based on weather pattern

## ======================================
## Test 12: Ch9, Brussels Potato, Ex4
## ======================================

years = 1985:2005
yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)
netirr_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)

for (year in years) {

    ## yield
    aos_v0_yield = get_aos_output(
        aosdir,
        "Output_Ch9_Brussels_Potato_Ex4",
        paste0("Exercise_9_4_NetIrr_", year, "_FinalOutput_orig.txt"),
        "Yield")

    aos_v5_yield = get_aos_output(
        aosdir,
        "Output_Ch9_Brussels_Potato_Ex4",
        paste0("Ch9_Brussels_Potato_Ex4_", year, "_FinalOutput.txt"),
        "Yield")

    aqpy_yield = 
        get_aqpy_output(
            aqpydir,
            paste0("Output_Ch9_Brussels_Potato_Ex4_", year),
            "Y_dailyTot_output.nc") %>%
        max %>% round(digits=2)

    ## net irr
    aos_v0_netirr = get_aos_output(
        aosdir,
        "Output_Ch9_Brussels_Potato_Ex4",
        paste0("Exercise_9_4_NetIrr_", year, "_WaterFluxes_orig.txt"),
        "NetIrrigation") %>% sum
        
    aos_v5_netirr = get_aos_output(
        aosdir,
        "Output_Ch9_Brussels_Potato_Ex4",
        paste0("Ch9_Brussels_Potato_Ex4_", year, "_WaterFluxes.txt"),
        "Irr") %>% sum

    aqpy_netirr =
        get_aqpy_output(
            aqpydir,
            paste0("Output_Ch9_Brussels_Potato_Ex4_", year),
            "IrrNetCum_dailyTot_output.nc") %>%
        max %>% round(digits=2)
    
    row = which(yield_comparison$year %in% year)
    yield_comparison$aos_v0[row] = aos_v0_yield
    yield_comparison$aos_v5[row] = aos_v5_yield
    yield_comparison$aqpy[row] = aqpy_yield

    netirr_comparison$aos_v0[row] = aos_v0_netirr
    netirr_comparison$aos_v5[row] = aos_v5_netirr
    netirr_comparison$aqpy[row] = aqpy_netirr
}

## yield
x = yield_comparison$aos_v5
y = yield_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p1 =
    data.frame(time=years, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Crop yield") +
    xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
    ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,15,2), limits=c(0,15)) +
    scale_y_continuous(breaks=seq(0,15,2), limits=c(0,15)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p1

## net irrigation
x = netirr_comparison$aos_v5
y = netirr_comparison$aqpy
rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
p2 =
    data.frame(time=years, x=x, y=y) %>%
    ggplot(aes(x,y)) +
    geom_point() +
    coord_fixed() +
    ggtitle("Net irrigation") +
    xlab('AquaCrop-OS (mm)') +
    ylab('AquaCrop-Py (mm)') +
    geom_abline(slope=1, intercept=c(0,0)) +
    scale_x_continuous(breaks=seq(0,500,100), limits=c(0,500)) +
    scale_y_continuous(breaks=seq(0,500,100), limits=c(0,500)) +
    ## annotate("text",x=0.25,y=9.75,label="hello")
    theme(plot.title=element_text(hjust=0.5))
p2

grid.arrange(p1,p2,ncol=2)

## ======================================
## Test 13: Ch9, Brussels Potato, Ex5
## ======================================

years = 1985:2005
aos_orig_suffix = c("RAW36","RAW100","RAW150")
suffix = c("Ex5a","Ex5b","Ex5c")
plot_titles = c("36% RAW","100% RAW","150% RAW")
yield_plots=list()
netirr_plots=list()

for (i in 1:length(suffix)) {    
    yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)
    netirr_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)

    for (j in 1:length(years)) {
        year = years[j]
        
        ## yield
        aos_v0_yield = get_aos_output(
            aosdir,
            paste0("Output_Ch9_Brussels_Potato_", suffix[i]),
            paste0("Exercise_9_5_", aos_orig_suffix[i], "_", year, "_FinalOutput_orig.txt"),
            "Yield")

        aos_v5_yield = get_aos_output(
            aosdir,
            paste0("Output_Ch9_Brussels_Potato_", suffix[i]),
            paste0("Ch9_Brussels_Potato_", suffix[i], "_", year, "_FinalOutput.txt"),
            "Yield")

        aqpy_yield = 
            get_aqpy_output(
                aqpydir,
                paste0("Output_Ch9_Brussels_Potato_", suffix[i], "_", year),
                "Y_dailyTot_output.nc") %>%
            max %>% round(digits=2)

        ## net irr
        aos_v0_netirr = get_aos_output(
            aosdir,
            paste0("Output_Ch9_Brussels_Potato_", suffix[i]),
            paste0("Exercise_9_5_", aos_orig_suffix[i], "_", year, "_WaterFluxes_orig.txt"),
            "NetIrrigation") %>% sum

        aos_v5_netirr = get_aos_output(
            aosdir,
            paste0("Output_Ch9_Brussels_Potato_", suffix[i]),
            paste0("Ch9_Brussels_Potato_", suffix[i], "_", year, "_WaterFluxes.txt"),
            "Irr") %>% sum

        aqpy_netirr =
            get_aqpy_output(
                aqpydir,
                paste0("Output_Ch9_Brussels_Potato_", suffix[i], "_", year),
                "IrrCum_dailyTot_output.nc") %>%
            max %>% round(digits=2)

        row = which(yield_comparison$year %in% year)
        yield_comparison$aos_v0[row] = aos_v0_yield
        yield_comparison$aos_v5[row] = aos_v5_yield
        yield_comparison$aqpy[row] = aqpy_yield

        netirr_comparison$aos_v0[row] = aos_v0_netirr
        netirr_comparison$aos_v5[row] = aos_v5_netirr
        netirr_comparison$aqpy[row] = aqpy_netirr
    }

    ## yield
    x = yield_comparison$aos_v5
    y = yield_comparison$aqpy
    rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
    rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
    p1 =
        data.frame(time=years, x=x, y=y) %>%
        ggplot(aes(x,y)) +
        geom_point() +
        coord_fixed() +
        ggtitle("Crop yield") +
        xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
        ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
        geom_abline(slope=1, intercept=c(0,0)) +
        scale_x_continuous(breaks=seq(0,15,2), limits=c(0,15)) +
        scale_y_continuous(breaks=seq(0,15,2), limits=c(0,15)) +
        ## annotate("text",x=0.25,y=9.75,label="hello")
        theme(plot.title=element_text(hjust=0.5))
    ## p1

    ## net irrigation
    x = netirr_comparison$aos_v5
    y = netirr_comparison$aqpy
    rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
    rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
    p2 =
        data.frame(time=years, x=x, y=y) %>%
        ggplot(aes(x,y)) +
        geom_point() +
        coord_fixed() +
        ggtitle("Net irrigation") +
        xlab('AquaCrop-OS (mm)') +
        ylab('AquaCrop-Py (mm)') +
        geom_abline(slope=1, intercept=c(0,0)) +
        scale_x_continuous(breaks=seq(0,500,100), limits=c(0,500)) +
        scale_y_continuous(breaks=seq(0,500,100), limits=c(0,500)) +
        ## annotate("text",x=0.25,y=9.75,label="hello")
        theme(plot.title=element_text(hjust=0.5))
    ## p2
    yield_plots[[i]] = p1
    netirr_plots[[i]] = p2
}

grid.arrange(yield_plots[[1]], netirr_plots[[1]],
             yield_plots[[2]], netirr_plots[[2]],
             yield_plots[[3]], netirr_plots[[3]], ncol=2)

## ======================================
## Test 14: Ch9, Brussels Potato, Ex6
## ======================================

yrs = list(c(1976:2005), c(2041:2070))
aos_orig_suffix = c("Hist","Fut")
suffix = c("Ex6a","Ex6b")
plot_titles = c("Historic","Future")
plots=list()
for (i in 1:length(suffix)) {
    
    years = yrs[[i]]
    yield_comparison = data.frame(year=years,aos_v0=NA,aos_v5=NA,aqpy=NA)

    for (j in 1:length(years)) {
        year = years[j]

        ## yield
        if (year > 2040) { yr = year - 2040 } else { yr = year }
        aos_v0_yield = get_aos_output(
            aosdir,
            paste0("Output_Ch9_Brussels_Potato_", suffix[i]),
            paste0("Exercise_9_6_", aos_orig_suffix[i], "_", yr, "_FinalOutput_orig.txt"),
            "Yield")

        aos_v5_yield = get_aos_output(
            aosdir,
            paste0("Output_Ch9_Brussels_Potato_", suffix[i]),
            paste0("Ch9_Brussels_Potato_", suffix[i], "_", year, "_FinalOutput.txt"),
            "Yield")

        aqpy_yield = 
            get_aqpy_output(
                aqpydir,
                paste0("Output_Ch9_Brussels_Potato_", suffix[i], "_", year),
                "Y_dailyTot_output.nc") %>%
            max %>% round(digits=2)

        row = which(yield_comparison$year %in% years[j])
        yield_comparison$aos_v0[row] = aos_v0_yield
        yield_comparison$aos_v5[row] = aos_v5_yield
        yield_comparison$aqpy[row] = aqpy_yield
    }

    ## yield
    x = yield_comparison$aos_v5
    y = yield_comparison$aqpy
    rmse = sqrt(mean((y-x)^2)) %>% round(digits=3)
    rsq = lm(y ~ x) %>% summary %>% `[[`("r.squared") %>% round(digits=3)
    p1 =
        data.frame(time=years, x=x, y=y) %>%
        ggplot(aes(x,y)) +
        geom_point() +
        coord_fixed() +
        ggtitle(plot_titles[i]) +
        ## ggtitle("Crop yield") +
        xlab(bquote('AquaCrop-OS ('*tonne~ha^-1*')')) +
        ylab(bquote('AquaCrop-Py ('*tonne~ha^-1*')')) +
        geom_abline(slope=1, intercept=c(0,0)) +
        scale_x_continuous(breaks=seq(0,15,2), limits=c(0,15)) +
        scale_y_continuous(breaks=seq(0,15,2), limits=c(0,15)) +
        ## annotate("text",x=0.25,y=9.75,label="hello")
        theme(plot.title=element_text(hjust=0.5))
    plots[[i]] = p1
}

p1 = plots[[1]]
p2 = plots[[2]]
grid.arrange(p1, p2, ncol=2)

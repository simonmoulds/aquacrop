## Author : Simon Moulds
## Date   : July 2018

library(stringr)
library(magrittr)

## Functions:

update_clock_input = function(x, ...) {
    x = x %>%
        str_replace_all("^(%%) (Simulation) (start|end) (time) (\\(yyyy-mm-dd-hh-mm-ss\\)) (%%)$", "\\1 \\2 \\3 \\4 \\(yyyy-mm-dd\\) \\6") %>%
        str_replace_all("^([ ]+)?(SimulationStartTime|SimulationEndTime)[ ]+:[ ]+([0-9]{4})-([0-9]{2})-([0-9]{2})-([0-9]{2})-([0-9]{2})-([0-9]{2})$", "\\1\\2 : \\3-\\4-\\5") %>%
        `[`(!grepl("^%% Time-step", .)) %>% 
        `[`(!grepl("^([ ]+)?TimeStep", .)) %>%
        `[`(!grepl("^%% Number of time-steps evaporation", .)) %>%
        `[`(!grepl("^([ ]+)?EvapTimeSteps", .)) %>%
        append("%% Simulate off-season ('N' or 'Y') %%") %>%
        append("OffSeason : Y")
    x
}

update_weather_input = function(x, ...) {
    ## header
    xh = c("%% ---------- Weather input time-series for AquaCropOS ---------- %%",
           "%% Day\tMonth\tYear\tMinTemp\tMaxTemp\tPrecipitation\tReferenceET %%")

    ## data
    ix = grep("^(\\s+)?[0-9]{2}/[0-9]{2}/[0-9]{4}", x) %>% sort
    nd = length(ix)
    xd = rep(NA, nd)
    d = x[ix] %>% str_replace("^\\s+|\\s+$","") %>% str_split("\t")
    for (i in 1:length(d)) {
        date = d[[i]][1] %>% str_split("/", simplify=TRUE) %>% as.integer %>% as.character
        data = d[[i]][2:length(d[[i]])] %>% as.numeric %>% as.character
        d[[i]] = c(date, data)
        xd[i] = paste0(d[[i]], collapse="\t")
    }

    x = c(xh, xd)
    x
}

update_initial_wc_input = function(x, ...) {
    x = x %>%
        str_replace_all("^([ ]+)?Type[ ]+:[ ]+", "") %>%
        str_replace_all("%% Interpolate data points ('Y' or 'N') %%", "%% Method ('Depth': Interpolate depth points; 'Layer': Constant value for each soil layer) %%") %>%
        str_replace_all("^([ ]+)?(Interp|Method)([ ]+)?:[ ]+Y", "Depth") %>%
        str_replace_all("^([ ]+)?(Interp|Method)([ ]+)?:[ ]+N", "Layer") %>%
        str_replace_all("^([ ]+)?(Interp|Method)([ ]+)?:[ ]+(Depth|Layer)", "\\4") %>%
        str_replace_all("^([ ]+)?NbrPts[ ]+:[ ]+", "")

    idx = grep("^%%\\s+Input data points", x)
    n = length(x) - idx
    if (n > 0) {
        ## d = as.data.frame(matrix(data=NA, nrow=n, ncol=2))
        for (i in 1:n) {            
            xi = idx + i
            x[xi] %<>% str_replace("\\s+$", "") %>% str_replace("\\s+", "\t")
            ## dr = x[xi] %>% str_replace("\\s+$", "") %>% str_split("\\s+", simplify=TRUE) %>% as.numeric ##%>% `[`(1:2)
            ## d[i,] = dr
        }
    }
    x
}

update_irr_mgmt_input = function(x, irr_schedule_fn="IrrigationSchedule.txt", ...) {

    add_irr_schedule = function(x, ...) {
        if (!any(grepl("Irrigation.*filename", x))) {
            x1 = x[1]
            x2 = x[2:length(x)]
            x1 = append(x1, "%% Irrigation time-series filename %%")
            x1 = append(x1, irr_schedule_fn)
            x = c(x1,x2)
        }
        x
    }

    x = x %>%
        add_irr_schedule %>%
        `[`(!grepl("^%%[ ]+Method for determining maximum irrigation depth", .)) %>%
        `[`(!grepl("MaxIrrMethod", .)) %>%
        `[`(!grepl("^%%[ ]+Well yield", .)) %>%
        `[`(!grepl("WellYield", .)) %>%
        `[`(!grepl("^%%[ ]+Irrigated area", .)) %>%
        `[`(!grepl("IrrArea", .)) %>%
        `[`(!grepl("^%%[ ]+Soil bund height", .)) %>%
        `[`(!grepl("zBund", .))
    x
}    

update_irr_schedule_input = function(x, year, ...) {

    if (any(grepl("%% Day\\\tMonth\\\tYear\\\tIrrigation\\(mm\\) %%", x))) {
        return(x)
    }
        
    ## header
    xh = c("%% -------- Irrigation schedule time-series for AquaCropOS -------- %%",
           "%% Day\tMonth\tYear\tIrrigation(mm) %%")

    ## data
    ix = grep("^(\\s+)?[0-9]{2}/[0-9]{2}/[0-9]{4}", x) %>% sort
    nd = length(ix)
    xd = rep(NA, nd)
    d = x[ix] %>% str_replace("^\\s+|\\s+$","") %>% str_split("\t")
    year0 = d[[1]][1] %>% str_split("/", simplify=TRUE) %>% `[`(1,3) %>% as.numeric
    diff = year0 - year
    
    for (i in 1:length(d)) {
        date = d[[i]][1] %>% str_split("/", simplify=TRUE) %>% as.integer %>% as.character
        date[3] = date[3] %>% as.numeric %>% `-`(diff) %>% as.character
        data = d[[i]][2:length(d[[i]])] %>% as.numeric %>% as.character
        d[[i]] = c(date, data)
        xd[i] = paste0(d[[i]], collapse="\t")
    }
    x = c(xh, xd)
    x
}

update_co2_input = function(x, ...) {
    x = x %>%
        str_replace_all("^[ ]+", "") %>%
        str_replace_all("^(\\d*)[ ]+(\\d*\\.?\\d*)$", "\\1    \\2")
    x
}    

update_field_mgmt_input = function(x, ...) {

    ## mulches
    mulchpct = x[grepl("^([ ]+)?mulchpct", x)]
    mulchpct_val = str_split(mulchpct, "\\s+:\\s+", simplify=TRUE)[2] %>% as.numeric
    mulches_val = ifelse(mulchpct_val > 0, 1, 0)
    fmulch = x[grepl("^(\\s+)?fmulch", x)]
    fmulch_val = str_split(fmulch, "\\s+:\\s+", simplify=TRUE)[2] %>% as.numeric

    ## bunds
    zbund = x[grepl("^(\\s+)?zBund", x)]
    zbund_val = str_split(zbund, "\\s+:\\s+", simplify=TRUE)[2] %>% as.numeric
    bunds_val = ifelse(zbund_val > 0, 1, 0)
    bundwater_val = 0 # can't find this in input files

    x = c("%% ---------- Soil parameter inputs for AquaCropOS ---------- %%",
          "%% Soil surface covered by mulches (0: No; 1: Yes) %%",
          paste0("Mulches : ", mulches_val),
          "%% Area of soil surface covered by mulches during growing season (%) %%",
          paste0("MulchPctGS : ", mulchpct_val),
          "%% Area of soil surface covered by mulches outside growing season (%) %%",
          paste0("MulchPctOS : ", mulchpct_val),
          "%% Soil evaporation adjustment factor due to effect of mulches %%",
          paste0("fMulch : ", fmulch_val),
          "%% Surface bunds present (0: No; 1: Yes) %%",
          paste0("Bunds : ", bunds_val),
          "%% Bund height (m) %%",
          paste0("zBund : ", zbund_val),
          "%% Initial water height in surface bunds (mm) %%",
          paste0("BundWater : ", bundwater_val)
          )
    x
}


update_crop_param_input = function(x, ...) {
    croptype = x[grepl("^(\\s+)?CropType", x)]
    calendartype = x[grepl("^(\\s+)?CalendarType", x)]
    switchgdd = "SwitchGDD : 1"
    plantingdate = x[grepl("^(\\s+)?PlantDate", x)] %>% str_replace("PlantDate","PlantingDate")
    harvestdate = x[grepl("^(\\s+)?HarvestDate", x)]
    emergence = x[grepl("^(\\s+)?Emergence", x)]
    maxrooting = x[grepl("^(\\s+)?MaxRooting", x)]
    senescence = x[grepl("^(\\s+)?Senescence", x)]
    maturity = x[grepl("^(\\s+)?Maturity", x)]
    histart = x[grepl("^(\\s+)?HIstart", x)]
    flowering = x[grepl("^(\\s+)?Flowering", x)]
    yldform = x[grepl("^(\\s+)?YldForm", x)]
    gddmethod = x[grepl("^(\\s+)?GDDmethod", x)]
    tbase = x[grepl("^(\\s+)?Tbase", x)]
    tupp = x[grepl("^(\\s+)?Tupp", x)]
    ## polheatstress = x[grepl("^(\\s+)?PolHeatStress", x)]
    polheatstress = "PolHeatStress : 1"
    tmax_up = x[grepl("^(\\s+)?Tmax_up", x)]
    tmax_lo = x[grepl("^(\\s+)?Tmax_lo", x)]
    ## polcoldstress = x[grepl("^(\\s+)?PolColdStress", x)]
    polcoldstress = "PolColdStress : 1"
    tmin_up = x[grepl("^(\\s+)?Tmin_up", x)]
    tmin_lo = x[grepl("^(\\s+)?Tmin_lo", x)]
    ## biotempstress = x[grepl("^(\\s+)?BioTempStress", x)]
    biotempstress = "BioTempStress : 1"
    gdd_up = x[grepl("^(\\s+)?GDD_up", x)]
    gdd_lo = x[grepl("^(\\s+)?GDD_lo", x)]
    fshape_b = x[grepl("^(\\s+)?fshape_b", x)]
    pctzmin = x[grepl("^(\\s+)?PctZmin", x)]
    zmin = x[grepl("^(\\s+)?Zmin", x)]
    zmax = x[grepl("^(\\s+)?Zmax", x)]
    fshape_r = x[grepl("^(\\s+)?fshape_r", x)]
    fshape_ex = x[grepl("^(\\s+)?fshape_ex", x)]
    sxtopq = x[grepl("^(\\s+)?SxTopQ", x)]
    sxbotq = x[grepl("^(\\s+)?SxBotQ", x)]
    a_tr = x[grepl("^(\\s+)?a_Tr", x)]
    seedsize = x[grepl("^(\\s+)?SeedSize", x)]
    plantpop = x[grepl("^(\\s+)?PlantPop", x)]
    ccmin = x[grepl("^(\\s+)?CCmin", x)]
    ccx = x[grepl("^(\\s+)?CCx", x)]
    cdc = x[grepl("^(\\s+)?CDC", x)]
    cgc = x[grepl("^(\\s+)?CGC", x)]
    kcb = x[grepl("^(\\s+)?Kcb", x)]
    fage = x[grepl("^(\\s+)?fage", x)]
    wp = x[grepl("^(\\s+)?WP\\s+", x)]
    ## wpy = x[grepl("^(\\s+)?WPy", x)]
    wpy = "WPy : 100"
    fsink = x[grepl("^(\\s+)?fsink", x)]
    bsted = x[grepl("^(\\s+)?bsted", x)]
    bface = x[grepl("^(\\s+)?bface", x)]
    hi0 = x[grepl("^(\\s+)?HI0", x)]
    hiini = x[grepl("^(\\s+)?HIini", x)]
    dhi_pre = x[grepl("^(\\s+)?dHI_pre", x)]
    a_hi = x[grepl("^(\\s+)?a_HI", x)]
    b_hi = x[grepl("^(\\s+)?b_HI", x)]
    dhi0 = x[grepl("^(\\s+)?dHI0", x)]
    determinant = x[grepl("^(\\s+)?Determinant", x)]
    exc = x[grepl("^(\\s+)?exc", x)]
    maxflowpct = x[grepl("^(\\s+)?MaxFlowPct", x)]
    p_up1 = x[grepl("^(\\s+)?p_up1", x)]
    p_up2 = x[grepl("^(\\s+)?p_up2", x)]
    p_up3 = x[grepl("^(\\s+)?p_up3", x)]
    p_up4 = x[grepl("^(\\s+)?p_up4", x)]
    p_lo1 = x[grepl("^(\\s+)?p_lo1", x)]
    p_lo2 = x[grepl("^(\\s+)?p_lo2", x)]
    p_lo3 = x[grepl("^(\\s+)?p_lo3", x)]
    p_lo4 = x[grepl("^(\\s+)?p_lo4", x)]
    fshape_w1 = x[grepl("^(\\s+)?fshape_w1", x)]
    fshape_w2 = x[grepl("^(\\s+)?fshape_w2", x)]
    fshape_w3 = x[grepl("^(\\s+)?fshape_w3", x)]
    fshape_w4 = x[grepl("^(\\s+)?fshape_w4", x)]
    etadj = x[grepl("^(\\s+)?etadj", x)] %>% str_replace("etadj", "ETadj")
    aer = x[grepl("^(\\s+)?aer", x)] %>% str_replace("aer","Aer")
    lagaer = x[grepl("^(\\s+)?LagAer", x)]
    beta = x[grepl("^(\\s+)?beta", x)]
    germthr = x[grepl("^(\\s+)?GermThr", x)]

    x = c("%% ---------- Crop parameters for AquaCropOS ---------- %%",
          "%% Crop Type ('1': Leafy vegetable, '2': Root/tuber, '3': Fruit/grain) %%",
          croptype,
          "%% Calendar Type ('1': Calendar days, '2': Growing degree days) %%",
          calendartype,
          "%% Convert calendar to GDD mode if inputs are given in calendar days ('0': No; '1': Yes) %%",
          switchgdd,
          "%% Planting Date (dd/mm) %%",
          plantingdate,
          "%% Latest Harvest Date (dd/mm) %%",
          harvestdate,
          "%% Growing degree/Calendar days from sowing to emergence/transplant recovery %%",
          emergence,
          "%% Growing degree/Calendar days from sowing to maximum rooting %%",
          maxrooting,
          "%% Growing degree/Calendar days from sowing to senescence %%",
          senescence,
          "%% Growing degree/Calendar days from sowing to maturity %%",
          maturity,
          "%% Growing degree/Calendar days from sowing to start of yield formation %%",
          histart,
          "%% Duration of flowering in growing degree/calendar days (-999 for non-fruit/grain crops) %%",
          flowering,
          "%% Duration of yield formation in growing degree/calendar days %%",
          yldform,
          "%% Growing degree day calculation method %%",
          gddmethod,
          "%% Base temperature (degC) below which growth does not progress %%",
          tbase,
          "%% Upper temperature (degC) above which crop development no longer increases %%",
          tupp,
          "%% Pollination affected by heat stress (0: No; 1: Yes) %%",
          polheatstress,
          "%% Maximum air temperature (degC) above which pollination begins to fail %%",
          tmax_up,
          "%% Maximum air temperature (degC) at which pollination completely fails %%",
          tmax_lo,
          "%% Pollination affected by cold stress (0: No; 1: Yes) %%",
          polcoldstress,
          "%% Minimum air temperature (degC) below which pollination begins to fail %%",
          tmin_up,
          "%% Minimum air temperature (degC) at which pollination completely fails %%",
          tmin_lo,
          "%% Biomass production affected by temperature stress (0: No; 1: Yes) %%",
          biotempstress,
          "%% Minimum growing degree days (degC/day) required for full biomass production %%",
          gdd_up,
          "%% Growing degree days (degC/day) at which no biomass production occurs %%",
          gdd_lo,
          "%% Shape factor describing the reduction in biomass production for insufficient growing degree days %%",
          fshape_b,
          "%% Initial percentage of minimum effective rooting depth %%",
          pctzmin,
          "%% Minimum effective rooting depth (m) %%",
          zmin,
          "%% Maximum rooting depth (m) %%",
          zmax,
          "%% Shape factor describing root expansion %%",
          fshape_r,
          "%% Shape factor describing the effects of water stress on root expansion %%",
          fshape_ex,
          "%% Maximum root water extraction at top of the root zone (m3/m3/day) %%",
          sxtopq,
          "%% Maximum root water extraction at the bottom of the root zone (m3/m3/day) %%",
          sxbotq,
          "%% Exponent parameter for adjustment of Kcx once senescence is triggered %%",
          a_tr,
          "%% Soil surface area (cm2) covered by an individual seedling at 90% emergence %%",
          seedsize,
          "%% Number of plants per hectare %%",
          plantpop,
          "%% Minimum canopy size below which yield formation cannot occur %%",
          ccmin,
          "%% Maximum canopy cover (fraction of soil cover) %%",
          ccx,
          "%% Canopy decline coefficient (fraction per GDD) %%",
          cdc,
          "%% Canopy growth coefficient (fraction per GDD) %%",
          cgc,
          "%% Crop coefficient when canopy growth is complete but prior to senescence %%",
          kcb,
          "%% Decline of crop coefficient due to ageing (%/day) %%",
          fage,
          "%% Water productivity normalized for ET0 and C02 (g/m2) %%",
          wp,
          "%% Adjustment of water productivity in yield formation stage (% of WP) %%",
          wpy,
          "%% Crop co2 sink strength coefficient %%",
          fsink,
          "%% WP co2 adjustment parameter given by Steduto et al. 2007 %%",
          bsted,
          "%% WP co2 adjustment parameter given by FACE experiments %%",
          bface,
          "%% Reference harvest index %%",
          hi0,
          "%% Initial harvest index %%",
          hiini,
          "%% Possible increase of harvest index due to water stress before flowering (%) %%",
          dhi_pre,
          "%% Coefficient describing positive impact on harvest index of restricted vegetative growth during yield formation %%",
          a_hi,
          "%% Coefficient describing negative impact on harvest index of stomatal closure during yield formation %%",
          b_hi,
          "%% Maximum allowable increase of harvest index above reference %%",
          dhi0,
          "%% Crop Determinancy ('0': Indeterminant, '1': Determinant) %%",
          determinant,
          "%% Excess of potential fruits %%",
          exc,
          "%% Percentage of total flowering at which peak flowering occurs %%",
          maxflowpct,
          "%% Upper soil water depletion threshold for water stress effects on affect canopy expansion %%",
          p_up1,
          "%% Upper soil water depletion threshold for water stress effects on canopy stomatal control %%",
          p_up2,
          "%% Upper soil water depletion threshold for water stress effects on canopy senescence %%",
          p_up3,
          "%% Upper soil water depletion threshold for water stress effects on canopy pollination %%",
          p_up4,
          "%% Lower soil water depletion threshold for water stress effects on canopy expansion %%",
          p_lo1,
          "%% Lower soil water depletion threshold for water stress effects on canopy stomatal control %%",
          p_lo2,
          "%% Lower soil water depletion threshold for water stress effects on canopy senescence %%",
          p_lo3,
          "%% Lower soil water depletion threshold for water stress effects on canopy pollination %%",
          p_lo4,
          "%% Shape factor describing water stress effects on canopy expansion %%",
          fshape_w1,
          "%% Shape factor describing water stress effects on stomatal control %%",
          fshape_w2,
          "%% Shape factor describing water stress effects on canopy senescence %%",
          fshape_w3,
          "%% Shape factor describing water stress effects on pollination %%",
          fshape_w4,
          "%% Adjustment to water stress thresholds depending on daily ET0 (0: 'No', 1: 'Yes') %%",
          etadj,
          "%% Vol (%) below saturation at which stress begins to occur due to deficient aeration %%",
          aer,
          "%% Number of days lag before aeration stress affects crop growth %%",
          lagaer,
          "%% Reduction (%) to p_lo3 when early canopy senescence is triggered %%",
          beta,
          "%% Proportion of total water storage needed for crop to germinate %%",
          germthr)
    x
}

update_soil_param_input = function(x, soil_profile_fn="SoilProfile.txt", soil_texture_fn="SoilTexture.txt", soil_hydrology_fn="SoilHydrology.txt") {
    calcshp = x[grepl("^(\\s+)?CalcSHP", x)]
    zsoil = x[grepl("^(\\s+)?Zsoil", x)] %>% str_replace("Zsoil","zSoil")
    ncomp = x[grepl("^(\\s+)?ncomp", x, ignore.case=TRUE)] %>% str_replace("ncomp","nComp")
    nlayer = x[grepl("^(\\s+)?nlayer", x, ignore.case=TRUE)] %>% str_replace("nlayer","nLayer")
    evapzsurf = x[grepl("^(\\s+)?EvapZsurf", x)]
    evapzmin = x[grepl("^(\\s+)?EvapZmin", x)]
    evapzmax = x[grepl("^(\\s+)?EvapZmax", x)]
    kex = x[grepl("^(\\s+)?Kex", x)]
    fevap = x[grepl("^(\\s+)?fevap", x)]
    fwrelexp = x[grepl("^(\\s+)?fWrelExp", x)]
    fwcc = x[grepl("^(\\s+)?fwcc", x)]
    ## adjrew = x[grepl("^(\\s+)?AdjREW", x)]
    adjrew = "AdjREW : 0"
    ## rew = x[grepl("^(\\s+)?fWrelExp", x)]
    rew = "REW : 9"
    adjcn = x[grepl("^(\\s+)?AdjCN", x)]
    cn = x[grepl("^(\\s+)?CN", x)]
    zcn = x[grepl("^(\\s+)?Zcn", x)] %>% str_replace("Zcn","zCN")
    zgerm = x[grepl("^(\\s+)?Zgerm", x)] %>% str_replace("Zgerm","zGerm")
    zres = x[grepl("^(\\s+)?zRes", x)]
    ## fshape_cr = x[grepl("^(\\s+)?fshape_cr", x)]
    fshape_cr = "fshape_cr : 16"

    x = c("%% ---------- Soil parameter inputs for AquaCropOS ---------- %%",
          "%% Soil profile filename %%",
          soil_profile_fn,
          "%% Soil textural properties filename %%",
          soil_texture_fn,
          "%% Soil hydraulic properties filename %%",
          soil_hydrology_fn,
          "%% Calculate soil hydraulic properties (0: No, 1: Yes) %%",
          calcshp,
          "%% Total thickness of soil profile (m) %%",
          zsoil,
          "%% Total number of compartments %%",
          ncomp,
          "%% Total number of layers %%",
          nlayer,
          "%% Thickness of soil surface skin evaporation layer (m) %%",
          evapzsurf,
          "%% Minimum thickness of full soil surface evaporation layer (m) %%",
          evapzmin,
          "%% Maximum thickness of full soil surface evaporation layer (m) %%",
          evapzmax,
          "%% Maximum soil evaporation coefficient %%",
          kex,
          "%% Shape factor describing reduction in soil evaporation %%",
          fevap,
          "%% Proportional value of Wrel at which soil evaporation layer expands %%",
          fwrelexp,
          "%% Maximum coefficient for soil evaporation reduction due to sheltering effect of withered canopy %%",
          fwcc,
          "%% Adjust default value for readily evaporable water (0: No, 1: Yes) %%",
          adjrew,
          "%% Readily evaporable water (mm) (only used if adjusting) %%",
          rew,      
          "%% Adjust curve number for antecedent moisture content (0:No, 1:Yes) %%",
          adjcn,
          "%% Curve number %%",
          cn,
          "%% Thickness of soil surface (m) used to calculate water content to adjust curve number %%",
          zcn,
          "%% Thickness of soil surface (m) used to calculate water content for germination %%",
          zgerm,
          "%% Depth of restrictive soil layer (set to negative value if not present) %%",
          zres,
          "%% Capillary rise shape factor %%",
          fshape_cr)
    x
}

update_soil_hydrology_input = function(x, layer_thicknesses, ...) {
    ix = grep("^(\\s+)?1\\s+", x) - 1
    nd = length(x) - ix
    ## xx = x[grep("%%", x, invert=TRUE)] %>% str_split("\\s+|\t")
    if (any(grepl("LayerThickness",x))) {
        layer_thicknesses=""
    } else {    
        if (nd != length(layer_thicknesses)) {
            stop("soil hydrology input does not correspond with soil profile")
        }
    }
    
    ## d = as.data.frame(matrix(data=NA, nrow=length(xx), ncol=length(xx[[1]]) + 1))
    for (i in 1:nd) {
        ii = ix + i
        d = x[ii] %>% str_replace("\\s+$", "") %>% str_split("\\s+|\t", simplify=TRUE) %>% as.character
        d = c(d[1], layer_thicknesses[i], d[2:length(d)])
        x[ii] = paste0(d, collapse="\t")
        ## xxi = xx[[i]]
        ## xxi = c(xxi[1], thickness[i], xxi[2:length(xxi)])
        ## d[i,] = xxi
        ## xx[[i]] = xxi
    }
    xd = x[-(1:ix)] %>% gsub("\t\t","\t",.)

    xh = c("%% ---------- Soil hydraulic properties for AquaCropOS ---------- %%",
           "%% LayerNo  LayerThickness(m)  thS(m3/m3)    thFC(m3/m3)  thWP(m3/m3)   Ksat(mm/day) %%")

    x = c(xh, xd)
    x
}

compute_layer_thickness = function(soil_profile, ...) {
    ix = grep("^(\\s+)?\\d+\\s+\\d+(\\.\\d+)?\\s+\\d+", soil_profile)
    soil_profile = soil_profile %>% str_replace("^\\s+|\\s+$", "")
    x = read.table(text=soil_profile[ix], colClasses=c("integer","double","integer")) %>% setNames(c("compartment","thickness","layer"))
    nlayer = max(x[["layer"]])
    thickness = rep(NA, nlayer)
    for (i in 1:nlayer) {
        ix = (x[["layer"]] == i)
        xx = x[ix,]
        thickness[i] = sum(xx[["thickness"]])
    }
    thickness
}

update_soil_profile_input = function(x, ...) {
    ix = grep("^(\\s+)?1\\s+", x)
    n = length(x)
    for (i in ix:n) {
        x[i] %<>% str_replace("\\s+$", "") %>% str_replace_all("\\s+", "\t")
    }
    xh = c("%% ---------- Soil profile discretisation for AquaCropOS ---------- %%",
           "%% CompartmentNo\tThickness(m)\tLayerNo %%")
    xd = x[ix:length(x)]
    x = append(xh, xd)
    x
}

update_crop_mix_input = function(ncrop, planting_calendar, crop_rotation_filename, crop_nm, crop_param_fn, irri_mgmt_fn) {
    x = c("%% ---------- Crop mix options for AquaCropOS ---------- %%",
          "%% Number of crop options %%",
          ncrop,
          "%% Specified planting calendar %%",
          planting_calendar,
          "%% Crop rotation filename %%",
          crop_rotation_filename,
          "%% Information about each crop type %%",
          "%% CropType, CropFilename, IrrigationFilename %%",
          paste(crop_nm, crop_param_fn, irri_mgmt_fn, sep=", "))
}

module aquacrop_w
  use types
  use aquacrop, only: update_aquacrop
  implicit none

contains

  subroutine update_aquacrop_w( &
       gdd, &
       gdd_cum, &
       gdd_method, &
       t_max, &
       t_min, &
       t_base, &
       t_upp, &
       growth_stage, &
       canopy_10pct, &
       max_canopy, &
       senescence, &
       dap, &
       delayed_cds, &
       delayed_gdds, &       
       th, &
       th_fc_adj, &
       wt_in_soil, &       
       th_s, &
       th_fc, &
       wt, &
       variable_wt, &
       zgw, &
       dz, &
       layer_ix, &       
       pre_irr, &
       irr_method, &
       z_root, &
       z_min, &
       net_irr_smt, &
       th_wilt, &
       dz_sum, &
       deep_perc, &
       flux_out, &
       k_sat, &
       tau, &
       runoff, &
       infl, &
       prec, &
       days_submrgd, &
       bund, &
       z_bund, &
       cn, &
       adj_cn, &
       z_cn, &
       cn_bot, &
       cn_top, &
       thrz_act, &
       thrz_sat, &
       thrz_fc, &
       thrz_wilt, &
       thrz_dry, &
       thrz_aer, &
       taw, &
       dr, &
       th_dry, &
       aer, &
       irr, &
       irr_cum, &
       irr_net_cum, &
       smt1, &
       smt2, &
       smt3, &
       smt4, &
       irr_scheduled, &
       app_eff, &
       et_ref, &
       max_irr, &
       irr_interval, &
       surf_stor, &
       cr_tot, &
       a_cr, &
       b_cr, &
       f_shape_cr, &
       dz_layer, &
       germ, &
       z_germ, &
       germ_thr, &
       r_cor, &
       z_max, &
       pct_z_min, &
       emergence, &
       max_rooting, &
       fshape_r, &
       fshape_ex, &
       sx_bot, &
       sx_top, &
       tr_ratio, &
       z_res, &
       cc, &
       cc_prev, &       
       cc_adj, &
       cc_ns, &
       cc_adj_ns, &
       ccx_w, &
       ccx_act, &
       ccx_w_ns, &
       ccx_act_ns, &       
       cc0_adj, &
       ccx_early_sen, &       
       t_early_sen, &
       premat_senes, &
       crop_dead, &
       cc0, &
       ccx, &
       cgc, &
       cdc, &
       maturity, &
       canopy_dev_end, &
       et_adj, &
       p_up1, &
       p_up2, &
       p_up3, &
       p_up4, &
       p_lo1, &
       p_lo2, &
       p_lo3, &
       p_lo4, &
       f_shape_w1, &
       f_shape_w2, &
       f_shape_w3, &
       f_shape_w4, &
       es_act, &
       e_pot, &
       wet_surf, &
       w_surf, &
       w_stage_two, &
       evap_z, &
       evap_z_min, &
       evap_z_max, &
       rew, &
       kex, &
       fwcc, &
       f_evap, &
       f_wrel_exp, &
       mulches, &
       f_mulch, &
       mulch_pct_gs, &
       mulch_pct_os, &
       time_step, &
       evap_time_steps, &
       tr_pot0, &
       tr_pot_ns, &
       tr_act, &
       tr_act0, &
       t_pot, &
       aer_days, &
       aer_days_comp, &       
       age_days, &
       age_days_ns, &
       day_submrgd, &
       irr_net, &
       max_canopy_cd, &
       kcb, &
       a_tr, &
       fage, &
       lag_aer, &
       co2_conc, &
       co2_refconc, &
       et_pot, &
       gw_in, &
       hi_ref, &
       pct_lag_phase, &
       yield_form, &
       cc_min, &
       hi_ini, &
       hi0, &
       higc, &
       hi_start, &
       hi_start_cd, &
       t_lin_switch, &
       dhi_linear, &
       crop_type, &
       bio_temp_stress, &
       gdd_up, &
       gdd_lo, &
       pol_heat_stress, &
       t_max_up, &
       t_max_lo, &
       f_shp_b, &
       pol_cold_stress, &
       t_min_up, &
       t_min_lo, &
       b, &
       b_ns, &
       yld_form_cd, &
       wp, &
       wpy, &
       f_co2, &
       determinant, &       
       hi_adj, &
       pre_adj, &
       f_pre, &
       f_pol, &
       f_post, &
       fpost_dwn, &
       fpost_upp, &
       s_cor1, &
       s_cor2, &
       dhi0, &
       dhi_pre, &
       canopy_dev_end_cd, &
       hi_end_cd, &
       flowering_cd, &
       a_hi, &
       b_hi, &
       exc, &
       yield, &
       crop_mature, &       
       calendar_type, &
       growing_season_day1, &
       growing_season, &
       n_farm, n_crop, n_comp, n_layer, n_cell &       
       )

    ! rearrange arguments:
    ! forcings (temp, prec, et, co2)
    ! state variables
    ! intermediate variables
    ! parameters
    ! - soil
    ! - crop
    ! - others
    ! options
    
    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: gdd ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: gdd_cum ! state variable
    
    integer(int32), intent(in) :: gdd_method             ! option
    
    real(real64), dimension(n_cell), intent(in) :: t_max ! input
    real(real64), dimension(n_cell), intent(in) :: t_min ! input
    
    real(real64), dimension(n_cell, n_crop), intent(in) :: t_base ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: t_upp  ! crop param

    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: growth_stage ! intermediate variable
    
    real(real64), dimension(n_cell, n_crop), intent(in) :: canopy_10pct ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: max_canopy ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: senescence ! crop param

    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: dap ! state variable
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: delayed_cds ! state variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: delayed_gdds ! state variable

    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: th ! state variable
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: th_fc_adj ! intermediate variable
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: wt_in_soil ! flag
    real(real64), dimension(n_cell, n_layer), intent(in) :: th_s ! soil hydraulic param
    real(real64), dimension(n_cell, n_layer), intent(in) :: th_fc ! soil hydraulic param
    integer(int32), intent(in) :: wt                              ! option
    integer(int32), intent(in) :: variable_wt                     ! option
    real(real64), dimension(n_cell), intent(in) :: zgw            ! input (from netcdf)
    real(real64), dimension(n_comp), intent(in) :: dz             ! input (from config)
    integer(int32), dimension(n_comp), intent(in) :: layer_ix     ! soil profile

    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: pre_irr ! intermediate variable
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: irr_method ! irrigation mgmt param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: z_root ! intermediate variable
    real(real64), dimension(n_cell, n_crop), intent(in) :: z_min ! crop param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: net_irr_smt ! irrigation mgmt param
    real(real64), dimension(n_cell, n_layer), intent(in) :: th_wilt ! soil hydraulic param
    real(real64), dimension(n_comp), intent(in) :: dz_sum           ! soil profile
    
    ! drainage
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: deep_perc ! intermediate variable
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: flux_out ! state variable
    real(real64), dimension(n_cell, n_layer), intent(in) :: k_sat ! soil hydraulic param
    real(real64), dimension(n_cell, n_layer), intent(in) :: tau   ! soil hydraulic param

    ! rainfall_partition
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: runoff ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: infl ! intermediate variable
    real(real64), dimension(n_cell), intent(in) :: prec ! input
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: days_submrgd ! state variable
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: bund ! field mgmt param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_bund ! field mgmt param
    integer(int32), dimension(n_cell), intent(in) :: cn ! soil param
    
    integer(int32), intent(in) :: adj_cn                ! option
    
    real(real64), dimension(n_cell), intent(in) :: z_cn ! soil param
    real(real64), dimension(n_cell), intent(in) :: cn_bot ! soil param
    real(real64), dimension(n_cell), intent(in) :: cn_top ! soil param

    ! root_zone_water
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_act ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_sat ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_fc ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_wilt ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_dry ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_aer ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: taw ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: dr ! intermediate variable
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_dry ! soil hydraulic param
    real(real64), dimension(n_cell, n_crop), intent(in) :: aer ! crop param
    ! irrigation
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr_cum ! state variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr_net_cum ! state variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt1 ! irrigation mgmt param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt2 ! irrigation mgmt param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt3 ! irrigation mgmt param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt4 ! irrigation mgmt param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr_scheduled ! irrigation mgmt param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: app_eff ! irrigation mgmt param
    real(real64), dimension(n_cell), intent(in) :: et_ref ! input
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: max_irr ! irrigation mgmt param
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: irr_interval ! irrigation mgmt param
    ! infiltration
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: surf_stor ! state variable
    ! capillary_rise
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cr_tot ! intermediate variable
    real(real64), dimension(n_cell, n_layer), intent(in) :: a_cr ! soil hydraulic param
    real(real64), dimension(n_cell, n_layer), intent(in) :: b_cr ! soil hydraulic param
    real(real64), dimension(n_cell), intent(in) :: f_shape_cr ! soil parameter
    real(real64), dimension(n_layer), intent(in) :: dz_layer  ! soil profile
    ! germination
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: germ ! state variable (?)
    real(real64), dimension(n_cell), intent(in) :: z_germ ! soil param
    real(real64), dimension(n_cell, n_crop), intent(in) :: germ_thr ! crop param
    ! root_development
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: r_cor ! state variable (?)
    real(real64), dimension(n_cell, n_crop), intent(in) :: z_max ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: pct_z_min ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: emergence ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: max_rooting ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: fshape_r ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: fshape_ex ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: sx_bot ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: sx_top ! crop param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: tr_ratio ! intermediate variable
    real(real64), dimension(n_cell), intent(in) :: z_res ! soil param
    ! canopy_cover
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc ! intermediate variable
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_prev ! state variable
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_adj ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_ns ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_adj_ns ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_w ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_act ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_w_ns ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_act_ns ! intermediate variable
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc0_adj ! state variable (?)    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_early_sen ! state variable (?)
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: t_early_sen ! state variable (?)
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: premat_senes ! state variable (?)
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: crop_dead ! state variable (?)

    real(real64), dimension(n_cell, n_crop), intent(in) :: cc0 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: ccx ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: cgc ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: cdc ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: maturity ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: canopy_dev_end ! crop param
    
    integer(int32), dimension(n_cell, n_crop), intent(in) :: et_adj ! crop param - should this be a switch???
    
    real(real64), dimension(n_cell, n_crop), intent(in) :: p_up1 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: p_up2 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: p_up3 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: p_up4 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: p_lo1 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: p_lo2 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: p_lo3 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: p_lo4 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: f_shape_w1 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: f_shape_w2 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: f_shape_w3 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: f_shape_w4 ! crop param

    ! soil_evaporation    
    integer(int32), intent(in) :: time_step ! clock
    integer(int32), intent(in) :: evap_time_steps ! option (default value = 20)
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: wet_surf ! irrigation mgmt param
    real(real64), dimension(n_cell), intent(in) :: rew ! soil param
    real(real64), dimension(n_cell), intent(in) :: evap_z_min ! soil param
    real(real64), dimension(n_cell), intent(in) :: evap_z_max ! soil param
    real(real64), dimension(n_cell), intent(in) :: kex        ! soil param
    real(real64), dimension(n_cell), intent(in) :: fwcc ! soil paran
    real(real64), dimension(n_cell), intent(in) :: f_evap ! soil param
    real(real64), dimension(n_cell), intent(in) :: f_wrel_exp ! soil param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_mulch ! field mgmt param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: mulch_pct_gs ! field mgmt param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: mulch_pct_os ! field mgmt param
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: mulches ! field mgmt param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: es_act ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: e_pot ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: w_surf ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: w_stage_two ! state variable (?)
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: evap_z ! state variable (?)
    
    ! transpiration
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: tr_pot0 ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: tr_pot_ns ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: tr_act ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: tr_act0 ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: t_pot ! intermediate variable
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: aer_days ! state variable
    integer(int32), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: aer_days_comp ! state variable
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: age_days ! state variable
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: age_days_ns ! state variable
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: day_submrgd ! state variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr_net ! intermediate variable
    integer(int32), dimension(n_cell, n_crop), intent(in) :: max_canopy_cd ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: kcb ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: a_tr ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: fage ! crop param
    integer(int32), dimension(n_cell, n_crop), intent(in) :: lag_aer ! crop param
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: co2_conc
    real(real64), intent(in) :: co2_refconc

    ! evapotranspiration
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: et_pot ! intermediate variable

    ! inflow
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: gw_in ! intermediate variable

    ! harvest_index
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: hi_ref ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: pct_lag_phase ! intermediate variable
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: yield_form ! intermediate variable
    real(real64), dimension(n_cell, n_crop), intent(in) :: cc_min ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: hi_ini ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: hi0 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: higc ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: hi_start ! crop param
    integer(int32), dimension(n_cell, n_crop), intent(in) :: hi_start_cd ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: t_lin_switch ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: dhi_linear ! crop param
    integer(int32), dimension(n_cell, n_crop), intent(in) :: crop_type ! crop param

    ! temperature_stress
    integer(int32), dimension(n_cell, n_crop), intent(in) :: bio_temp_stress ! crop param
    integer(int32), dimension(n_cell, n_crop), intent(in) :: pol_heat_stress ! crop param
    integer(int32), dimension(n_cell, n_crop), intent(in) :: pol_cold_stress ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: gdd_up, gdd_lo ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: t_max_up, t_max_lo ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: t_min_up, t_min_lo ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: f_shp_b ! crop param

    ! biomass_accumulation
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: b ! state variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: b_ns ! state variable
    integer(int32), dimension(n_cell, n_crop), intent(in) :: yld_form_cd ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: wp ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: wpy ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: f_co2 ! crop param
    integer(int32), dimension(n_cell, n_crop), intent(in) :: determinant ! crop param

    ! adjust_harvest_index
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: hi_adj ! state variable
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: pre_adj ! state variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_pre ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_pol ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_post ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: fpost_dwn ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: fpost_upp ! intermediate variable
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: s_cor1 ! state variable (?)
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: s_cor2 ! state variable (?)
    real(real64), dimension(n_cell, n_crop), intent(in) :: dhi0 ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: dhi_pre ! crop param
    integer(int32), dimension(n_cell, n_crop), intent(in) :: canopy_dev_end_cd ! crop param
    integer(int32), dimension(n_cell, n_crop), intent(in) :: hi_end_cd ! crop param
    integer(int32), dimension(n_cell, n_crop), intent(in) :: flowering_cd ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: a_hi ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: b_hi ! crop param
    real(real64), dimension(n_cell, n_crop), intent(in) :: exc ! crop param
    
    ! crop_yield
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: yield ! intermediate variable
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: crop_mature ! state variable (?)
    
    integer(int32), intent(in) :: calendar_type ! option
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season_day1
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season
    integer(int32) :: i, j, k
    
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_aquacrop( &
                  gdd(k,j,i), &
                  gdd_cum(k,j,i), &
                  gdd_method, &
                  t_max(k), &
                  t_min(k), &
                  t_base(k,j), &
                  t_upp(k,j), &
                  growth_stage(k,j,i), &
                  canopy_10pct(k,j), &
                  max_canopy(k,j), &
                  senescence(k,j), &
                  dap(k,j,i), &
                  delayed_cds(k,j,i), &
                  delayed_gdds(k,j,i), &
                  th(k,:,j,i), &
                  th_fc_adj(k,:,j,i), &
                  wt_in_soil(k,j,i), &       
                  th_s(k,:), &
                  th_fc(k,:), &
                  wt, &
                  variable_wt, &
                  zgw(k), &
                  dz, &
                  layer_ix, &
                  pre_irr(k,j,i), &
                  irr_method(k,j,i), &
                  z_root(k,j,i), &
                  z_min(k,j), &
                  net_irr_smt(k,j,i), &
                  th_wilt(k,:), &
                  dz_sum, &
                  deep_perc(k,j,i), &
                  flux_out(k,:,j,i), &
                  k_sat(k,:), &
                  tau(k,:), &
                  runoff(k,j,i), &
                  infl(k,j,i), &
                  prec(k), &
                  days_submrgd(k,j,i), &
                  bund(k,j,i), &
                  z_bund(k,j,i), &
                  cn(k), &
                  adj_cn, &
                  z_cn(k), &
                  cn_bot(k), &
                  cn_top(k), &
                  thrz_act(k,j,i), &
                  thrz_sat(k,j,i), &
                  thrz_fc(k,j,i), &
                  thrz_wilt(k,j,i), &
                  thrz_dry(k,j,i), &
                  thrz_aer(k,j,i), &
                  taw(k,j,i), &
                  dr(k,j,i), &
                  th_dry(k,:,j,i), &
                  aer(k,j), &
                  irr(k,j,i), &
                  irr_cum(k,j,i), &
                  irr_net_cum(k,j,i), &
                  smt1(k,j,i), &
                  smt2(k,j,i), &
                  smt3(k,j,i), &
                  smt4(k,j,i), &
                  irr_scheduled(k,j,i), &
                  app_eff(k,j,i), &
                  et_ref(k), &
                  max_irr(k,j,i), &
                  irr_interval(k,j,i), &
                  surf_stor(k,j,i), &
                  cr_tot(k,j,i), &
                  a_cr(k,:), &
                  b_cr(k,:), &
                  f_shape_cr(k), &
                  dz_layer, &
                  germ(k,j,i), &
                  z_germ(k), &
                  germ_thr(k,j), &
                  r_cor(k,j,i), &
                  z_max(k,j), &
                  pct_z_min(k,j), &
                  emergence(k,j), &
                  max_rooting(k,j), &
                  fshape_r(k,j), &
                  fshape_ex(k,j), &
                  sx_bot(k,j), &
                  sx_top(k,j), &
                  tr_ratio(k,j,i), &
                  z_res(k), &
                  cc(k,j,i), &
                  cc_prev(k,j,i), &
                  cc_adj(k,j,i), &
                  cc_ns(k,j,i), &
                  cc_adj_ns(k,j,i), &
                  ccx_w(k,j,i), &
                  ccx_act(k,j,i), &
                  ccx_w_ns(k,j,i), &
                  ccx_act_ns(k,j,i), &       
                  cc0_adj(k,j,i), &
                  ccx_early_sen(k,j,i), &       
                  t_early_sen(k,j,i), &
                  premat_senes(k,j,i), &
                  crop_dead(k,j,i), &
                  cc0(k,j), &
                  ccx(k,j), &
                  cgc(k,j), &
                  cdc(k,j), &
                  maturity(k,j), &
                  canopy_dev_end(k,j), &
                  et_adj(k,j), &
                  p_up1(k,j), &
                  p_up2(k,j), &
                  p_up3(k,j), &
                  p_up4(k,j), &
                  p_lo1(k,j), &
                  p_lo2(k,j), &
                  p_lo3(k,j), &
                  p_lo4(k,j), &
                  f_shape_w1(k,j), &
                  f_shape_w2(k,j), &
                  f_shape_w3(k,j), &
                  f_shape_w4(k,j), &
                  es_act(k,j,i), &
                  e_pot(k,j,i), &
                  wet_surf(k,j,i), &
                  w_surf(k,j,i), &
                  w_stage_two(k,j,i), &
                  evap_z(k,j,i), &
                  evap_z_min(k), &
                  evap_z_max(k), &
                  rew(k), &
                  kex(k), &
                  fwcc(k), &
                  f_evap(k), &
                  f_wrel_exp(k), &
                  mulches(k,j,i), &
                  f_mulch(k,j,i), &
                  mulch_pct_gs(k,j,i), &
                  mulch_pct_os(k,j,i), &
                  time_step, &
                  evap_time_steps, &
                  tr_pot0(k,j,i), &
                  tr_pot_ns(k,j,i), &
                  tr_act(k,j,i), &
                  tr_act0(k,j,i), &
                  t_pot(k,j,i), &
                  aer_days(k,j,i), &
                  aer_days_comp(k,:,j,i), &
                  age_days(k,j,i), &
                  age_days_ns(k,j,i), &
                  day_submrgd(k,j,i), &
                  irr_net(k,j,i), &
                  max_canopy_cd(k,j), &
                  kcb(k,j), &
                  a_tr(k,j), &
                  fage(k,j), &
                  lag_aer(k,j), &
                  co2_conc(k,j,i), &
                  co2_refconc, &
                  et_pot(k,j,i), &
                  gw_in(k,j,i), &
                  hi_ref(k,j,i), &
                  pct_lag_phase(k,j,i), &
                  yield_form(k,j,i), &
                  cc_min(k,j), &
                  hi_ini(k,j), &
                  hi0(k,j), &
                  higc(k,j), &
                  hi_start(k,j), &
                  hi_start_cd(k,j), &
                  t_lin_switch(k,j), &
                  dhi_linear(k,j), &
                  crop_type(k,j), &                  
                  bio_temp_stress(k,j), &
                  gdd_up(k,j), &
                  gdd_lo(k,j), &
                  pol_heat_stress(k,j), &
                  t_max_up(k,j), &
                  t_max_lo(k,j), &
                  f_shp_b(k,j), &
                  pol_cold_stress(k,j), &
                  t_min_up(k,j), &
                  t_min_lo(k,j), &
                  b(k,j,i), &
                  b_ns(k,j,i), &
                  yld_form_cd(k,j), &
                  wp(k,j), &
                  wpy(k,j), &
                  f_co2(k,j), &
                  determinant(k,j), &
                  hi_adj(k,j,i), &
                  pre_adj(k,j,i), &
                  f_pre(k,j,i), &
                  f_pol(k,j,i), &
                  f_post(k,j,i), &
                  fpost_dwn(k,j,i), &
                  fpost_upp(k,j,i), &
                  s_cor1(k,j,i), &
                  s_cor2(k,j,i), &
                  dhi0(k,j), &
                  dhi_pre(k,j), &
                  canopy_dev_end_cd(k,j), &
                  hi_end_cd(k,j), &
                  flowering_cd(k,j), &
                  a_hi(k,j), &
                  b_hi(k,j), &
                  exc(k,j), &                  
                  yield(k,j,i), &
                  crop_mature(k,j,i), &
                  calendar_type, &
                  growing_season_day1(k,j,i), &
                  growing_season(k,j,i) &
                  )

                  ! hi_adj(k,j,i), &
                  ! pre_adj(k,j,i), &
                  ! f_pre(k,j,i), &
                  ! f_pol(k,j,i), &
                  ! f_post(k,j,i), &
                  ! fpost_dwn(k,j,i), &
                  ! fpost_upp(k,j,i), &
                  ! s_cor1(k,j,i), &
                  ! s_cor2(k,j,i), &
                  ! dhi0(k,j,i), &
                  ! dhi_pre(k,j,i), &
                  ! canopy_dev_end_cd(k,j,i), &
                  ! hi_end_cd(k,j,i), &
                  ! flowering_cd(k,j,i), &
                  ! a_hi(k,j,i), &
                  ! b_hi(k,j,i), &
                  ! exc(k,j,i), &                  
             
          end do
       end do
    end do
  end subroutine update_aquacrop_w
end module aquacrop_w

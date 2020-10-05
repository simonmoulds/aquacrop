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

    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: gdd ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: gdd_cum ! state variable
    
    integer(int32), intent(in) :: gdd_method             ! option
    
    real(real64), dimension(n_cell), intent(in) :: t_max ! input
    real(real64), dimension(n_cell), intent(in) :: t_min ! input
    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_base ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_upp  ! crop param

    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: growth_stage ! intermediate variable
    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: canopy_10pct ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: max_canopy ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: senescence ! crop param

    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: dap ! state variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: delayed_cds ! state variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: delayed_gdds ! state variable

    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: th ! state variable
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: th_fc_adj ! intermediate variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: wt_in_soil ! flag
    real(real64), dimension(n_layer, n_cell), intent(in) :: th_s ! soil hydraulic param
    real(real64), dimension(n_layer, n_cell), intent(in) :: th_fc ! soil hydraulic param
    integer(int32), intent(in) :: wt                              ! option
    integer(int32), intent(in) :: variable_wt                     ! option
    real(real64), dimension(n_cell), intent(in) :: zgw            ! input (from netcdf)
    real(real64), dimension(n_comp), intent(in) :: dz             ! input (from config)
    integer(int32), dimension(n_comp), intent(in) :: layer_ix     ! soil profile

    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: pre_irr ! intermediate variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: irr_method ! irrigation mgmt param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: z_root ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_min ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: net_irr_smt ! irrigation mgmt param
    real(real64), dimension(n_layer, n_cell), intent(in) :: th_wilt ! soil hydraulic param
    real(real64), dimension(n_comp), intent(in) :: dz_sum           ! soil profile
    
    ! drainage
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: deep_perc ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: flux_out ! state variable
    real(real64), dimension(n_layer, n_cell), intent(in) :: k_sat ! soil hydraulic param
    real(real64), dimension(n_layer, n_cell), intent(in) :: tau   ! soil hydraulic param

    ! rainfall_partition
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: runoff ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: infl ! intermediate variable
    real(real64), dimension(n_cell), intent(in) :: prec ! input
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: days_submrgd ! state variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: bund ! field mgmt param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_bund ! field mgmt param
    integer(int32), dimension(n_cell), intent(in) :: cn ! soil param
    
    integer(int32), intent(in) :: adj_cn                ! option
    
    real(real64), dimension(n_cell), intent(in) :: z_cn ! soil param
    real(real64), dimension(n_cell), intent(in) :: cn_bot ! soil param
    real(real64), dimension(n_cell), intent(in) :: cn_top ! soil param

    ! root_zone_water
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_act ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_sat ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_fc ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_wilt ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_dry ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_aer ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: taw ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: dr ! intermediate variable
    real(real64), dimension(n_layer, n_cell), intent(in) :: th_dry ! soil hydraulic param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: aer ! crop param
    ! irrigation
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: irr ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: irr_cum ! state variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: irr_net_cum ! state variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: smt1 ! irrigation mgmt param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: smt2 ! irrigation mgmt param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: smt3 ! irrigation mgmt param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: smt4 ! irrigation mgmt param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: irr_scheduled ! irrigation mgmt param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: app_eff ! irrigation mgmt param
    real(real64), dimension(n_cell), intent(in) :: et_ref ! input
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: max_irr ! irrigation mgmt param
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: irr_interval ! irrigation mgmt param
    ! infiltration
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: surf_stor ! state variable
    ! capillary_rise
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cr_tot ! intermediate variable
    real(real64), dimension(n_layer, n_cell), intent(in) :: a_cr ! soil hydraulic param
    real(real64), dimension(n_layer, n_cell), intent(in) :: b_cr ! soil hydraulic param
    real(real64), dimension(n_cell), intent(in) :: f_shape_cr ! soil parameter
    real(real64), dimension(n_layer), intent(in) :: dz_layer  ! soil profile
    ! germination
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: germ ! state variable (?)
    real(real64), dimension(n_cell), intent(in) :: z_germ ! soil param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: germ_thr ! crop param
    ! root_development
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: r_cor ! state variable (?)
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_max ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: pct_z_min ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: emergence ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: max_rooting ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: fshape_r ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: fshape_ex ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: sx_bot ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: sx_top ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_ratio ! intermediate variable
    real(real64), dimension(n_cell), intent(in) :: z_res ! soil param
    ! canopy_cover
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc ! intermediate variable
    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc_prev ! state variable
    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc_adj ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc_ns ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc_adj_ns ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ccx_w ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ccx_act ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ccx_w_ns ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ccx_act_ns ! intermediate variable
    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc0_adj ! state variable (?)    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ccx_early_sen ! state variable (?)
    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: t_early_sen ! state variable (?)
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: premat_senes ! state variable (?)
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: crop_dead ! state variable (?)

    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc0 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ccx ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cgc ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cdc ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: maturity ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: canopy_dev_end ! crop param
    
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: et_adj ! crop param - should this be a switch???
    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up1 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up2 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up3 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up4 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo1 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo2 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo3 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo4 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w1 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w2 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w3 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w4 ! crop param

    ! soil_evaporation    
    integer(int32), intent(in) :: time_step ! clock
    integer(int32), intent(in) :: evap_time_steps ! option (default value = 20)
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: wet_surf ! irrigation mgmt param
    real(real64), dimension(n_cell), intent(in) :: rew ! soil param
    real(real64), dimension(n_cell), intent(in) :: evap_z_min ! soil param
    real(real64), dimension(n_cell), intent(in) :: evap_z_max ! soil param
    real(real64), dimension(n_cell), intent(in) :: kex        ! soil param
    real(real64), dimension(n_cell), intent(in) :: fwcc ! soil paran
    real(real64), dimension(n_cell), intent(in) :: f_evap ! soil param
    real(real64), dimension(n_cell), intent(in) :: f_wrel_exp ! soil param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_mulch ! field mgmt param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: mulch_pct_gs ! field mgmt param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: mulch_pct_os ! field mgmt param
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: mulches ! field mgmt param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: es_act ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: e_pot ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: w_surf ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: w_stage_two ! state variable (?)
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: evap_z ! state variable (?)
    
    ! transpiration
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_pot0 ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_pot_ns ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_act ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_act0 ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: t_pot ! intermediate variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: aer_days ! state variable
    integer(int32), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: aer_days_comp ! state variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: age_days ! state variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: age_days_ns ! state variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: day_submrgd ! state variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: irr_net ! intermediate variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: max_canopy_cd ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: kcb ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: a_tr ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: fage ! crop param
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: lag_aer ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: co2_conc
    real(real64), intent(in) :: co2_refconc

    ! evapotranspiration
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: et_pot ! intermediate variable

    ! inflow
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: gw_in ! intermediate variable

    ! harvest_index
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: hi_ref ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: pct_lag_phase ! intermediate variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: yield_form ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc_min ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_ini ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi0 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: higc ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start ! crop param
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start_cd ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_lin_switch ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: dhi_linear ! crop param
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: crop_type ! crop param

    ! temperature_stress
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: bio_temp_stress ! crop param
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: pol_heat_stress ! crop param
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: pol_cold_stress ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: gdd_up, gdd_lo ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_max_up, t_max_lo ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_min_up, t_min_lo ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shp_b ! crop param

    ! biomass_accumulation
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: b ! state variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: b_ns ! state variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: yld_form_cd ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: wp ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: wpy ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_co2 ! crop param
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: determinant ! crop param

    ! adjust_harvest_index
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: hi_adj ! state variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: pre_adj ! state variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: f_pre ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: f_pol ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: f_post ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: fpost_dwn ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: fpost_upp ! intermediate variable
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: s_cor1 ! state variable (?)
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: s_cor2 ! state variable (?)
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: dhi0 ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: dhi_pre ! crop param
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: canopy_dev_end_cd ! crop param
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_end_cd ! crop param
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: flowering_cd ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: a_hi ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: b_hi ! crop param
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: exc ! crop param
    
    ! crop_yield
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: yield ! intermediate variable
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: crop_mature ! state variable (?)
    
    integer(int32), intent(in) :: calendar_type ! option
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season_day1
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season
    integer(int32) :: i, j, k

    ! do i = 1, n_farm
    !    do j = 1, n_crop
    !       do k = 1, n_cell
    do i = 1, n_cell
       do j = 1, n_crop
          do k = 1, n_farm
             call update_aquacrop( &
                  gdd(k,j,i), &
                  gdd_cum(k,j,i), &
                  gdd_method, &
                  t_max(i), &
                  t_min(i), &
                  t_base(k,j,i), &
                  t_upp(k,j,i), &
                  growth_stage(k,j,i), &
                  canopy_10pct(k,j,i), &
                  max_canopy(k,j,i), &
                  senescence(k,j,i), &
                  dap(k,j,i), &
                  delayed_cds(k,j,i), &
                  delayed_gdds(k,j,i), &
                  th(k,j,:,i), &
                  th_fc_adj(k,j,:,i), &
                  wt_in_soil(k,j,i), &       
                  th_s(:,i), &
                  th_fc(:,i), &
                  wt, &
                  variable_wt, &
                  zgw(i), &
                  dz, &
                  layer_ix, &
                  pre_irr(k,j,i), &
                  irr_method(k,j,i), &
                  z_root(k,j,i), &
                  z_min(k,j,i), &
                  net_irr_smt(k,j,i), &
                  th_wilt(:,i), &
                  dz_sum, &
                  deep_perc(k,j,i), &
                  flux_out(k,j,:,i), &
                  k_sat(:,i), &
                  tau(:,i), &
                  runoff(k,j,i), &
                  infl(k,j,i), &
                  prec(i), &
                  days_submrgd(k,j,i), &
                  bund(k,j,i), &
                  z_bund(k,j,i), &
                  cn(i), &
                  adj_cn, &
                  z_cn(i), &
                  cn_bot(i), &
                  cn_top(i), &
                  thrz_act(k,j,i), &
                  thrz_sat(k,j,i), &
                  thrz_fc(k,j,i), &
                  thrz_wilt(k,j,i), &
                  thrz_dry(k,j,i), &
                  thrz_aer(k,j,i), &
                  taw(k,j,i), &
                  dr(k,j,i), &
                  th_dry(:,i), &
                  aer(k,j,i), &
                  irr(k,j,i), &
                  irr_cum(k,j,i), &
                  irr_net_cum(k,j,i), &
                  smt1(k,j,i), &
                  smt2(k,j,i), &
                  smt3(k,j,i), &
                  smt4(k,j,i), &
                  irr_scheduled(k,j,i), &
                  app_eff(k,j,i), &
                  et_ref(i), &
                  max_irr(k,j,i), &
                  irr_interval(k,j,i), &
                  surf_stor(k,j,i), &
                  cr_tot(k,j,i), &
                  a_cr(:,i), &
                  b_cr(:,i), &
                  f_shape_cr(i), &
                  dz_layer, &
                  germ(k,j,i), &
                  z_germ(i), &
                  germ_thr(k,j,i), &
                  r_cor(k,j,i), &
                  z_max(k,j,i), &
                  pct_z_min(k,j,i), &
                  emergence(k,j,i), &
                  max_rooting(k,j,i), &
                  fshape_r(k,j,i), &
                  fshape_ex(k,j,i), &
                  sx_bot(k,j,i), &
                  sx_top(k,j,i), &
                  tr_ratio(k,j,i), &
                  z_res(i), &
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
                  cc0(k,j,i), &
                  ccx(k,j,i), &
                  cgc(k,j,i), &
                  cdc(k,j,i), &
                  maturity(k,j,i), &
                  canopy_dev_end(k,j,i), &
                  et_adj(k,j,i), &
                  p_up1(k,j,i), &
                  p_up2(k,j,i), &
                  p_up3(k,j,i), &
                  p_up4(k,j,i), &
                  p_lo1(k,j,i), &
                  p_lo2(k,j,i), &
                  p_lo3(k,j,i), &
                  p_lo4(k,j,i), &
                  f_shape_w1(k,j,i), &
                  f_shape_w2(k,j,i), &
                  f_shape_w3(k,j,i), &
                  f_shape_w4(k,j,i), &
                  es_act(k,j,i), &
                  e_pot(k,j,i), &
                  wet_surf(k,j,i), &
                  w_surf(k,j,i), &
                  w_stage_two(k,j,i), &
                  evap_z(k,j,i), &
                  evap_z_min(i), &
                  evap_z_max(i), &
                  rew(i), &
                  kex(i), &
                  fwcc(i), &
                  f_evap(i), &
                  f_wrel_exp(i), &
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
                  aer_days_comp(k,j,:,i), &
                  age_days(k,j,i), &
                  age_days_ns(k,j,i), &
                  day_submrgd(k,j,i), &
                  irr_net(k,j,i), &
                  max_canopy_cd(k,j,i), &
                  kcb(k,j,i), &
                  a_tr(k,j,i), &
                  fage(k,j,i), &
                  lag_aer(k,j,i), &
                  co2_conc(k,j,i), &
                  co2_refconc, &
                  et_pot(k,j,i), &
                  gw_in(k,j,i), &
                  hi_ref(k,j,i), &
                  pct_lag_phase(k,j,i), &
                  yield_form(k,j,i), &
                  cc_min(k,j,i), &
                  hi_ini(k,j,i), &
                  hi0(k,j,i), &
                  higc(k,j,i), &
                  hi_start(k,j,i), &
                  hi_start_cd(k,j,i), &
                  t_lin_switch(k,j,i), &
                  dhi_linear(k,j,i), &
                  crop_type(k,j,i), &                  
                  bio_temp_stress(k,j,i), &
                  gdd_up(k,j,i), &
                  gdd_lo(k,j,i), &
                  pol_heat_stress(k,j,i), &
                  t_max_up(k,j,i), &
                  t_max_lo(k,j,i), &
                  f_shp_b(k,j,i), &
                  pol_cold_stress(k,j,i), &
                  t_min_up(k,j,i), &
                  t_min_lo(k,j,i), &
                  b(k,j,i), &
                  b_ns(k,j,i), &
                  yld_form_cd(k,j,i), &
                  wp(k,j,i), &
                  wpy(k,j,i), &
                  f_co2(k,j,i), &
                  determinant(k,j,i), &
                  hi_adj(k,j,i), &
                  pre_adj(k,j,i), &
                  f_pre(k,j,i), &
                  f_pol(k,j,i), &
                  f_post(k,j,i), &
                  fpost_dwn(k,j,i), &
                  fpost_upp(k,j,i), &
                  s_cor1(k,j,i), &
                  s_cor2(k,j,i), &
                  dhi0(k,j,i), &
                  dhi_pre(k,j,i), &
                  canopy_dev_end_cd(k,j,i), &
                  hi_end_cd(k,j,i), &
                  flowering_cd(k,j,i), &
                  a_hi(k,j,i), &
                  b_hi(k,j,i), &
                  exc(k,j,i), &                  
                  yield(k,j,i), &
                  crop_mature(k,j,i), &
                  calendar_type, &
                  growing_season_day1(k,j,i), &
                  growing_season(k,j,i) &
                  )
          end do
       end do
    end do
  end subroutine update_aquacrop_w
end module aquacrop_w

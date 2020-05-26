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
       ccxw, &
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
       ccxw_ns, &
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

    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: gdd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: gdd_cum
    integer(int32), intent(in) :: gdd_method
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: t_max
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: t_min
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: t_base
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: t_upp

    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: growth_stage
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: canopy_10pct
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: max_canopy
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: senescence
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: dap
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: delayed_cds
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: delayed_gdds

    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: th
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: th_fc_adj
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: wt_in_soil
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_s
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_fc    
    integer(int32), intent(in) :: wt
    integer(int32), intent(in) :: variable_wt
    real(real64), dimension(n_cell), intent(in) :: zgw
    real(real64), dimension(n_comp), intent(in) :: dz
    integer(int32), dimension(n_comp), intent(in) :: layer_ix

    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: pre_irr
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: irr_method
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: z_root
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_min
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: net_irr_smt
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_wilt
    real(real64), dimension(n_comp), intent(in) :: dz_sum
    ! drainage
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: deep_perc
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: flux_out
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: k_sat
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: tau
    ! rainfall_partition
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: runoff 
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: infl
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: prec
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: days_submrgd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: bund
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_bund
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: cn
    integer(int32), intent(in) :: adj_cn
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_cn
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cn_bot
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cn_top
    ! root_zone_water
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_act
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_sat
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_fc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_wilt
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_dry
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_aer
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: taw
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: dr
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_dry
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: aer
    ! irrigation
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr_cum
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr_net_cum
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt1
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt2
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt3
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt4
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr_scheduled    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: app_eff
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: et_ref
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: max_irr    
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: irr_interval
    ! infiltration
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: surf_stor
    ! capillary_rise
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cr_tot
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: a_cr
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: b_cr
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_shape_cr
    real(real64), dimension(n_layer), intent(in) :: dz_layer
    ! germination
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: germ
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_germ
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: germ_thr
    ! root_development
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: r_cor
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_max
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: pct_z_min
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: emergence
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: max_rooting
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: fshape_r
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: fshape_ex
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: sx_bot
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: sx_top
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: tr_ratio
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_res
    ! canopy_cover
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_prev
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_adj
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_adj_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_w
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_act
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_w_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_act_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc0_adj
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_early_sen    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: t_early_sen
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: premat_senes
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: crop_dead
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc0
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: ccx
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cgc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cdc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: maturity
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: canopy_dev_end
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: et_adj
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_up1
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_up2
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_up3
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_up4
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_lo1
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_lo2
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_lo3
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_lo4
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_shape_w1
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_shape_w2
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_shape_w3
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_shape_w4        
    ! soil_evaporation
    integer(int32), intent(in) :: time_step
    integer(int32), intent(in) :: evap_time_steps
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: wet_surf
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: rew
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: evap_z_min
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: evap_z_max    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: kex
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: ccxw
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: fwcc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_evap
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_wrel_exp
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_mulch
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: mulch_pct_gs
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: mulch_pct_os   
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: mulches
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: es_act
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: e_pot
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: w_surf
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: w_stage_two
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: evap_z
    ! transpiration
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: tr_pot0
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: tr_pot_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: tr_act
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: tr_act0
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: t_pot
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: aer_days
    integer(int32), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: aer_days_comp
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: age_days
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: age_days_ns
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: day_submrgd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr_net
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: max_canopy_cd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: kcb
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: ccxw_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: a_tr
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: fage
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: lag_aer
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: co2_conc
    real(real64), intent(in) :: co2_refconc
    ! evapotranspiration
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: et_pot
    ! inflow
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: gw_in
    ! harvest_index
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: hi_ref
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: pct_lag_phase
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: yield_form
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc_min
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_ini
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi0
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: higc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_start
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_start_cd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: t_lin_switch
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: dhi_linear
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: crop_type
    ! temperature_stress
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: bio_temp_stress    
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: pol_heat_stress
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: pol_cold_stress
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: gdd_up, gdd_lo
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: t_max_up, t_max_lo
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: t_min_up, t_min_lo
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_shp_b
    ! biomass_accumulation
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: b
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: b_ns
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: yld_form_cd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: wp
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: wpy
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_co2
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: determinant        
    ! adjust_harvest_index
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: hi_adj
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: pre_adj
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_pre
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_pol
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_post
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: fpost_dwn
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: fpost_upp
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: s_cor1
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: s_cor2
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: dhi0
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: dhi_pre
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: canopy_dev_end_cd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_end_cd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: flowering_cd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: a_hi
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: b_hi
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: exc
    ! crop_yield
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: yield
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: crop_mature
    
    integer(int32), intent(in) :: calendar_type
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
                  t_max(k,j,i), &
                  t_min(k,j,i), &
                  t_base(k,j,i), &
                  t_upp(k,j,i), &
                  growth_stage(k,j,i), &
                  canopy_10pct(k,j,i), &
                  max_canopy(k,j,i), &
                  senescence(k,j,i), &
                  dap(k,j,i), &
                  delayed_cds(k,j,i), &
                  delayed_gdds(k,j,i), &
                  th(k,:,j,i), &
                  th_fc_adj(k,:,j,i), &
                  wt_in_soil(k,j,i), &       
                  th_s(k,:,j,i), &
                  th_fc(k,:,j,i), &
                  wt, &
                  variable_wt, &
                  zgw(k), &
                  dz, &
                  layer_ix, &
                  pre_irr(k,j,i), &
                  irr_method(k,j,i), &
                  z_root(k,j,i), &
                  z_min(k,j,i), &
                  net_irr_smt(k,j,i), &
                  th_wilt(k,:,j,i), &
                  dz_sum, &
                  deep_perc(k,j,i), &
                  flux_out(k,:,j,i), &
                  k_sat(k,:,j,i), &
                  tau(k,:,j,i), &
                  runoff(k,j,i), &
                  infl(k,j,i), &
                  prec(k,j,i), &
                  days_submrgd(k,j,i), &
                  bund(k,j,i), &
                  z_bund(k,j,i), &
                  cn(k,j,i), &
                  adj_cn, &
                  z_cn(k,j,i), &
                  cn_bot(k,j,i), &
                  cn_top(k,j,i), &
                  thrz_act(k,j,i), &
                  thrz_sat(k,j,i), &
                  thrz_fc(k,j,i), &
                  thrz_wilt(k,j,i), &
                  thrz_dry(k,j,i), &
                  thrz_aer(k,j,i), &
                  taw(k,j,i), &
                  dr(k,j,i), &
                  th_dry(k,:,j,i), &
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
                  et_ref(k,j,i), &
                  max_irr(k,j,i), &
                  irr_interval(k,j,i), &
                  surf_stor(k,j,i), &
                  cr_tot(k,j,i), &
                  a_cr(k,:,j,i), &
                  b_cr(k,:,j,i), &
                  f_shape_cr(k,j,i), &
                  dz_layer, &
                  germ(k,j,i), &
                  z_germ(k,j,i), &
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
                  z_res(k,j,i), &
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
                  evap_z_min(k,j,i), &
                  evap_z_max(k,j,i), &
                  rew(k,j,i), &
                  kex(k,j,i), &
                  ccxw(k,j,i), &
                  fwcc(k,j,i), &
                  f_evap(k,j,i), &
                  f_wrel_exp(k,j,i), &
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
                  max_canopy_cd(k,j,i), &
                  kcb(k,j,i), &
                  ccxw_ns(k,j,i), &
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

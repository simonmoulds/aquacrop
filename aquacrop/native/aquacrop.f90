module aquacrop
  use types
  use gdd, only: update_gdd
  use growth_stage, only: update_growth_stage
  use check_gw_table, only: update_check_gw_table
  use pre_irr, only: update_pre_irr
  use drainage, only: update_drainage
  use rainfall_partition, only: update_rain_part
  use root_zone_water, only: update_root_zone_water
  use irrigation, only: update_irrigation
  use infiltration, only: update_infl
  use capillary_rise, only: update_cap_rise
  use germination, only: update_germ
  use root_dev, only: update_root_dev
  use canopy_cover, only: update_canopy_cover
  use soil_evaporation, only: update_soil_evap
  use water_stress, only: update_water_stress
  use transpiration, only: update_transpiration
  use inflow, only: update_inflow
  use harvest_index, only: update_harvest_index, adjust_harvest_index
  use temperature_stress, only: update_temp_stress
  use biomass_accumulation, only: update_biomass_accum
  use crop_yield, only: update_crop_yield
  implicit none

contains

  subroutine update_aquacrop( &
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
       irr_net_cum, &           ! TODO - move to pre_irr
       smt1, &                  ! soil moisture threshold
       smt2, &                  ! soil moisture threshold
       smt3, &                  ! soil moisture threshold
       smt4, &                  ! soil moisture threshold
       irr_scheduled, &
       app_eff, &               ! irrigation application efficiency
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
       growing_season &
       )
    
    ! from gdd.f90
    real(real64), intent(inout) :: gdd
    real(real64), intent(inout) :: gdd_cum
    ! from growth_stage.f90
    integer(int32), intent(inout) :: growth_stage
    ! from check_gw_table
    real(real64), dimension(:), intent(inout) :: th
    real(real64), dimension(:), intent(inout) :: th_fc_adj
    integer(int32), intent(inout) :: wt_in_soil    
    ! from gdd.f90
    integer(int32), intent(in) :: gdd_method
    real(real64), intent(in) :: t_max
    real(real64), intent(in) :: t_min
    real(real64), intent(in) :: t_base
    real(real64), intent(in) :: t_upp
    ! from growth_stage.f90
    real(real64), intent(in) :: canopy_10pct
    real(real64), intent(in) :: max_canopy
    real(real64), intent(in) :: senescence
    integer(int32), intent(in) :: dap
    integer(int32), intent(inout) :: delayed_cds
    real(real64), intent(inout) :: delayed_gdds
    integer(int32), intent(in) :: calendar_type
    integer(int32), intent(in) :: growing_season_day1
    integer(int32), intent(in) :: growing_season
    ! from check_gw_table
    real(real64), dimension(:), intent(in) :: th_s
    real(real64), dimension(:), intent(in) :: th_fc    
    integer(int32), intent(in) :: wt
    integer(int32), intent(in) :: variable_wt
    real(real64), intent(in) :: zgw
    real(real64), dimension(:), intent(in) :: dz
    integer(int32), dimension(:), intent(in) :: layer_ix
    ! from pre_irr
    real(real64), intent(inout) :: pre_irr
    integer(int32), intent(in) :: irr_method
    real(real64), intent(inout) :: z_root
    real(real64), intent(in) :: z_min
    real(real64), intent(in) :: net_irr_smt
    real(real64), dimension(:), intent(in) :: th_wilt
    real(real64), dimension(:), intent(in) :: dz_sum
    ! from drainage
    real(real64), intent(inout) :: deep_perc
    real(real64), dimension(:), intent(inout) :: flux_out
    real(real64), dimension(:), intent(in) :: k_sat
    real(real64), dimension(:), intent(in) :: tau
    ! from rainfall_partition
    real(real64), intent(inout) :: runoff 
    real(real64), intent(inout) :: infl
    real(real64), intent(in) :: prec
    integer(int32), intent(inout) :: days_submrgd
    integer(int32), intent(in) :: bund
    real(real64), intent(in) :: z_bund
    integer(int32), intent(in) :: cn
    integer(int32), intent(in) :: adj_cn
    real(real64), intent(in) :: z_cn
    real(real64), intent(in) :: cn_bot
    real(real64), intent(in) :: cn_top
    ! root_zone_water
    real(real64), intent(inout) :: thrz_act
    real(real64), intent(inout) :: thrz_sat
    real(real64), intent(inout) :: thrz_fc
    real(real64), intent(inout) :: thrz_wilt
    real(real64), intent(inout) :: thrz_dry
    real(real64), intent(inout) :: thrz_aer
    real(real64), intent(inout) :: taw
    real(real64), intent(inout) :: dr
    real(real64), dimension(:), intent(in) :: th_dry
    real(real64), intent(in) :: aer
    ! irrigation
    real(real64), intent(inout) :: irr
    real(real64), intent(inout) :: irr_cum
    real(real64), intent(inout) :: irr_net_cum    
    real(real64), intent(in) :: smt1
    real(real64), intent(in) :: smt2
    real(real64), intent(in) :: smt3
    real(real64), intent(in) :: smt4
    real(real64), intent(inout) :: irr_scheduled    
    real(real64), intent(in) :: app_eff
    real(real64), intent(in) :: et_ref
    real(real64), intent(in) :: max_irr    
    integer(int32), intent(in) :: irr_interval    
    ! infiltration
    real(real64), intent(inout) :: surf_stor
    ! capillary_rise
    real(real64), intent(inout) :: cr_tot
    real(real64), dimension(:), intent(in) :: a_cr
    real(real64), dimension(:), intent(in) :: b_cr
    real(real64), intent(in) :: f_shape_cr
    real(real64), dimension(:), intent(in) :: dz_layer
    ! germination
    integer(int32), intent(inout) :: germ
    real(real64), intent(in) :: z_germ
    real(real64), intent(in) :: germ_thr
    ! root_development
    real(real64), intent(inout) :: r_cor
    real(real64), intent(in) :: z_max
    real(real64), intent(in) :: pct_z_min
    real(real64), intent(in) :: emergence
    real(real64), intent(in) :: max_rooting
    real(real64), intent(in) :: fshape_r
    real(real64), intent(in) :: fshape_ex
    real(real64), intent(in) :: sx_bot
    real(real64), intent(in) :: sx_top
    real(real64), intent(inout) :: tr_ratio
    real(real64), intent(in) :: z_res
    ! canopy cover
    real(real64), intent(inout) :: cc
    real(real64), intent(inout) :: cc_prev
    real(real64), intent(inout) :: cc_adj
    real(real64), intent(inout) :: cc_ns
    real(real64), intent(inout) :: cc_adj_ns
    real(real64), intent(inout) :: ccx_w
    real(real64), intent(inout) :: ccx_act
    real(real64), intent(inout) :: ccx_w_ns
    real(real64), intent(inout) :: ccx_act_ns
    real(real64), intent(inout) :: cc0_adj
    real(real64), intent(inout) :: ccx_early_sen
    real(real64), intent(inout) :: t_early_sen
    integer(int32), intent(inout) :: premat_senes
    integer(int32), intent(inout) :: crop_dead
    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cgc
    real(real64), intent(in) :: cdc
    real(real64), intent(in) :: maturity
    real(real64), intent(in) :: canopy_dev_end
    integer(int32), intent(in) :: et_adj
    real(real64), intent(in) :: p_up1
    real(real64), intent(in) :: p_up2
    real(real64), intent(in) :: p_up3
    real(real64), intent(in) :: p_up4
    real(real64), intent(in) :: p_lo1
    real(real64), intent(in) :: p_lo2
    real(real64), intent(in) :: p_lo3
    real(real64), intent(in) :: p_lo4
    real(real64), intent(in) :: f_shape_w1
    real(real64), intent(in) :: f_shape_w2
    real(real64), intent(in) :: f_shape_w3
    real(real64), intent(in) :: f_shape_w4    
    ! soil evaporation
    integer(int32), intent(in) :: time_step
    integer(int32), intent(in) :: evap_time_steps
    real(real64), intent(in) :: wet_surf
    real(real64), intent(in) :: rew
    real(real64), intent(in) :: evap_z_min
    real(real64), intent(in) :: evap_z_max    
    real(real64), intent(in) :: kex
    ! real(real64), intent(in) :: ccxw
    real(real64), intent(in) :: fwcc
    real(real64), intent(in) :: f_evap
    real(real64), intent(in) :: f_wrel_exp
    real(real64), intent(in) :: f_mulch
    real(real64), intent(in) :: mulch_pct_gs
    real(real64), intent(in) :: mulch_pct_os    
    integer(int32), intent(in) :: mulches
    real(real64), intent(inout) :: es_act
    real(real64), intent(inout) :: e_pot
    real(real64), intent(inout) :: w_surf
    real(real64), intent(inout) :: w_stage_two
    real(real64), intent(inout) :: evap_z    
    ! transpiration
    real(real64), intent(inout) :: tr_pot0
    real(real64), intent(inout) :: tr_pot_ns
    real(real64), intent(inout) :: tr_act
    real(real64), intent(inout) :: tr_act0
    real(real64), intent(inout) :: t_pot
    integer(int32), intent(inout) :: aer_days
    integer(int32), dimension(:), intent(inout) :: aer_days_comp
    integer(int32), intent(inout) :: age_days
    integer(int32), intent(inout) :: age_days_ns
    integer(int32), intent(inout) :: day_submrgd
    real(real64), intent(inout) :: irr_net
    integer(int32), intent(in) :: max_canopy_cd
    real(real64), intent(in) :: kcb
    ! real(real64), intent(in) :: ccxw_ns
    real(real64), intent(in) :: a_tr
    real(real64), intent(in) :: fage
    integer(int32), intent(in) :: lag_aer
    real(real64), intent(in) :: co2_conc
    real(real64), intent(in) :: co2_refconc
    ! evapotranspiration
    real(real64), intent(inout) :: et_pot
    ! inflow
    real(real64), intent(inout) :: gw_in
    ! harvest_index
    real(real64), intent(inout) :: hi_ref
    real(real64), intent(inout) :: pct_lag_phase
    integer(int32), intent(inout) :: yield_form
    real(real64), intent(in) :: cc_min
    real(real64), intent(in) :: hi_ini
    real(real64), intent(in) :: hi0
    real(real64), intent(in) :: higc
    real(real64), intent(in) :: hi_start
    integer(int32), intent(in) :: hi_start_cd
    real(real64), intent(in) :: t_lin_switch
    real(real64), intent(in) :: dhi_linear
    integer(int32), intent(in) :: crop_type
    ! temperature_stress
    integer(int32), intent(in) :: bio_temp_stress    
    integer(int32), intent(in) :: pol_heat_stress
    integer(int32), intent(in) :: pol_cold_stress
    real(real64), intent(in) :: gdd_up, gdd_lo
    real(real64), intent(in) :: t_max_up, t_max_lo
    real(real64), intent(in) :: t_min_up, t_min_lo
    real(real64), intent(in) :: f_shp_b
    ! real(real64), intent(inout) :: kst_bio, kst_polh, kst_polc
    ! biomass_accumulation
    real(real64), intent(inout) :: b
    real(real64), intent(inout) :: b_ns
    integer(int32), intent(in) :: yld_form_cd
    real(real64), intent(in) :: wp
    real(real64), intent(in) :: wpy
    real(real64), intent(in) :: f_co2
    integer(int32), intent(in) :: determinant        
    ! adjust harvest index
    real(real64), intent(inout) :: hi_adj
    integer(int32), intent(inout) :: pre_adj
    real(real64), intent(inout) :: f_pre
    real(real64), intent(inout) :: f_pol
    real(real64), intent(inout) :: f_post
    real(real64), intent(inout) :: fpost_dwn
    real(real64), intent(inout) :: fpost_upp
    real(real64), intent(inout) :: s_cor1
    real(real64), intent(inout) :: s_cor2
    real(real64), intent(in) :: dhi0
    real(real64), intent(in) :: dhi_pre
    integer(int32), intent(in) :: canopy_dev_end_cd
    integer(int32), intent(in) :: hi_end_cd
    integer(int32), intent(in) :: flowering_cd
    real(real64), intent(in) :: a_hi
    real(real64), intent(in) :: b_hi
    real(real64), intent(in) :: exc
    ! crop_yield
    real(real64), intent(inout) :: yield
    integer(int32), intent(inout) :: crop_mature
    
    real(real64) :: ksw_exp, ksw_sto, ksw_sen, ksw_pol, ksw_stolin
    real(real64) :: kst_bio, kst_polh, kst_polc
    integer(int32) :: beta
    
    call update_gdd( &
         gdd, &
         gdd_cum, &
         gdd_method, &
         t_max, &
         t_min, &
         t_base, &
         t_upp, &
         growing_season &
         )

    call update_growth_stage( &
         growth_stage, &
         canopy_10pct, &
         max_canopy, &
         senescence, &
         gdd_cum, &
         dap, &
         delayed_cds, &
         delayed_gdds, &       
         calendar_type, &
         growing_season &
         )

    call update_check_gw_table( &
         th, &
         th_fc_adj, &
         wt_in_soil, &       
         th_s, &
         th_fc, &
         wt, &
         variable_wt, &
         zgw, &
         dz, &
         layer_ix &
         )

    call update_pre_irr( &
         pre_irr, &
         th, &
         irr_method, &
         dap, &
         z_root, &
         z_min, &
         net_irr_smt, &
         th_fc, &
         th_wilt, &
         dz, &
         dz_sum, &
         layer_ix &
         )

    call update_drainage( &
         th, &
         deep_perc, &
         flux_out, &
         th_s, &
         th_fc, &
         k_sat, &
         tau, &
         th_fc_adj, &
         dz, &
         dz_sum, &
         layer_ix &
         )
         
    call update_rain_part( &
         runoff, &
         infl, &
         prec, &
         th, &
         days_submrgd, &
         bund, &
         z_bund, &
         th_fc, &
         th_wilt, &
         cn, &
         adj_cn, &
         z_cn, &
         cn_bot, &
         cn_top, &
         dz, &
         dz_sum, &
         layer_ix &
         )

    call update_root_zone_water( &
         thrz_act, &
         thrz_sat, &
         thrz_fc, &
         thrz_wilt, &
         thrz_dry, &
         thrz_aer, &
         taw, &
         dr, &
         th, &
         th_s, &
         th_fc, &
         th_wilt, &
         th_dry, &
         aer, &
         z_root, &
         z_min, &
         dz, &
         dz_sum, &
         layer_ix &
         )

    call update_irrigation( &
         irr_method, &
         irr, &
         irr_cum, &
         irr_net_cum, &         ! TODO: move to pre_irr
         smt1, &
         smt2, &
         smt3, &
         smt4, &
         irr_scheduled, &
         app_eff, &
         z_root, &
         z_min, &
         taw, &
         dr, &
         thrz_fc, &
         thrz_act, &
         prec, &
         runoff, &
         et_ref, &
         max_irr, &
         irr_interval, &
         dap, &
         growth_stage, &
         growing_season_day1, &
         growing_season &
         )    

    call update_infl( &
         infl, &
         surf_stor, &
         flux_out, &
         deep_perc, &
         runoff, &
         th, &
         irr, &
         app_eff, &
         bund, &
         z_bund, &
         th_s, &
         th_fc, &
         th_fc_adj, &       
         k_sat, &
         tau, &
         dz, &
         layer_ix &
         )

    call update_cap_rise( &
         cr_tot, &
         th, &
         th_wilt, &
         th_fc, &
         th_fc_adj, &
         k_sat, &
         a_cr, &
         b_cr, &
         f_shape_cr, &
         flux_out, &
         wt, &
         zgw, &
         dz, &
         dz_layer, &
         layer_ix &
         )

    call update_germ( &
         germ, &
         delayed_cds, &
         delayed_gdds, &
         gdd, &
         th, &
         th_fc, &
         th_wilt, &
         z_germ, &
         germ_thr, &
         dz, &
         dz_sum, &
         layer_ix, &
         growing_season &
         )

    call update_root_dev( &
         z_root, &
         r_cor, &
         z_min, &
         z_max, &
         pct_z_min, &
         emergence, &
         max_rooting, &
         fshape_r, &
         fshape_ex, &
         sx_bot, &
         sx_top, &
         dap, &
         gdd, &
         gdd_cum, &
         delayed_cds, &
         delayed_gdds, &
         tr_ratio, &
         germ, &
         z_res, &
         wt, &
         zgw, &
         calendar_type, &
         growing_season_day1, &
         growing_season &
         )

    call update_root_zone_water( &
         thrz_act, &
         thrz_sat, &
         thrz_fc, &
         thrz_wilt, &
         thrz_dry, &
         thrz_aer, &
         taw, &
         dr, &
         th, &
         th_s, &
         th_fc, &
         th_wilt, &
         th_dry, &
         aer, &
         z_root, &
         z_min, &
         dz, &
         dz_sum, &
         layer_ix &
         )

    call update_canopy_cover( &
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
         gdd, &
         gdd_cum, &       
         cc0, &
         ccx, &
         cgc, &
         cdc, &
         emergence, &
         maturity, &
         senescence, &
         canopy_dev_end, &
         dr, &
         taw, &
         et_ref, &
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
         growing_season, &
         growing_season_day1, &
         calendar_type, &
         dap, &
         delayed_cds, &
         delayed_gdds &
         )

    call update_soil_evap( &
         prec, &
         et_ref, &
         es_act, &
         e_pot, &
         irr, &
         irr_method, &
         infl, &
         th, &
         th_s, &
         th_fc, &
         th_wilt, &
         th_dry, &
         surf_stor, &
         wet_surf, &
         w_surf, &
         w_stage_two, &
         cc, &
         cc_adj, &
         ccx_act, &
         evap_z, &
         evap_z_min, &
         evap_z_max, &
         rew, &
         kex, &
         ccx_w, &
         fwcc, &
         f_evap, &
         f_wrel_exp, &
         dz, &
         dz_sum, &
         mulches, &
         f_mulch, &
         mulch_pct_gs, &
         mulch_pct_os, &
         growing_season, &
         senescence, &
         premat_senes, &
         calendar_type, &
         dap, &
         gdd_cum, &
         delayed_cds, &
         delayed_gdds, &
         time_step, &
         evap_time_steps, &
         layer_ix &         
         )

    call update_root_zone_water( &
         thrz_act, &
         thrz_sat, &
         thrz_fc, &
         thrz_wilt, &
         thrz_dry, &
         thrz_aer, &
         taw, &
         dr, &
         th, &
         th_s, &
         th_fc, &
         th_wilt, &
         th_dry, &
         aer, &
         z_root, &
         z_min, &
         dz, &
         dz_sum, &
         layer_ix &
         )

    beta = 1
    call update_water_stress( &
         ksw_exp, &
         ksw_sto, &
         ksw_sen, &
         ksw_pol, &
         ksw_stolin, &
         dr, &
         taw, &
         et_ref, &
         et_adj, &
         t_early_sen, &
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
         beta &
         )

    call update_transpiration( &
         tr_pot0, &
         tr_pot_ns, &
         tr_act, &
         tr_act0, &
         t_pot, &
         tr_ratio, &
         aer_days, &
         aer_days_comp, &
         th, &
         thrz_act, &
         thrz_sat, &
         thrz_fc, &
         thrz_wilt, &
         thrz_dry, &
         thrz_aer, &
         taw, &
         dr, &
         age_days, &
         age_days_ns, &
         day_submrgd, &
         surf_stor, &
         irr_net, &
         irr_net_cum, &
         cc, &
         et_ref, &
         th_s, &
         th_fc, &
         th_wilt, &
         th_dry, &
         max_canopy_cd, &
         kcb, &
         ksw_stolin, &
         cc_adj, &
         cc_adj_ns, &
         cc_prev, &
         ccx_w, &
         ccx_w_ns, &
         z_root, &
         r_cor, &
         z_min, &
         a_tr, &
         aer, &
         fage, &
         lag_aer, &
         sx_bot, &
         sx_top, &
         et_adj, &
         p_lo2, &
         p_up2, &
         f_shape_w2, &
         irr_method, &
         net_irr_smt, &
         co2_conc, &
         co2_refconc, &
         dap, &
         delayed_cds, &
         dz, &
         dz_sum, &
         layer_ix, &
         growing_season_day1, &
         growing_season &         ! logical indicating whether the growing season is active
         )

    ! add this to transpiration routine?
    et_pot = e_pot + et_pot

    call update_inflow( &
         gw_in, &
         th, &
         wt, &
         zgw, &
         th_s, &
         dz, &
         layer_ix &
         )

    call update_harvest_index( &
         hi_ref, &
         pct_lag_phase, &
         yield_form, &
         cc_prev, &
         cc_min, &
         ccx, &
         hi_ini, &
         hi0, &
         higc, &
         hi_start, &
         hi_start_cd, &
         t_lin_switch, &
         dhi_linear, &
         gdd_cum, &
         dap, &
         delayed_cds, &
         delayed_gdds, &
         crop_type, &
         calendar_type, &
         growing_season &
         )

    call update_temp_stress( &
         bio_temp_stress, &
         kst_bio, &
         gdd, &
         gdd_up, &
         gdd_lo, &
         pol_heat_stress, &
         kst_polh, &
         t_max, &
         t_max_up, &
         t_max_lo, &
         f_shp_b, &
         pol_cold_stress, &
         kst_polc, &
         t_min, &
         t_min_up, &
         t_min_lo &
         )

    call update_biomass_accum( &
         et_ref, &
         tr_act, &
         tr_pot_ns, &
         b, &
         b_ns, &
         bio_temp_stress, &
         gdd, &
         gdd_up, &
         gdd_lo, &
         pol_heat_stress, &
         t_max, &
         t_max_up, &
         t_max_lo, &
         f_shp_b, &
         pol_cold_stress, &
         t_min, &
         t_min_up, &
         t_min_lo, &       
         hi_ref, &
         pct_lag_phase, &
         yld_form_cd, &
         wp, &
         wpy, &
         f_co2, &
         hi_start_cd, &
         delayed_cds, &
         dap, &
         crop_type, &
         determinant, &
         growing_season &
         )

    call update_root_zone_water( &
         thrz_act, &
         thrz_sat, &
         thrz_fc, &
         thrz_wilt, &
         thrz_dry, &
         thrz_aer, &
         taw, &
         dr, &
         th, &
         th_s, &
         th_fc, &
         th_wilt, &
         th_dry, &
         aer, &
         z_root, &
         z_min, &
         dz, &
         dz_sum, &
         layer_ix &
         )

    beta = 1
    call update_water_stress( &
         ksw_exp, &
         ksw_sto, &
         ksw_sen, &
         ksw_pol, &
         ksw_stolin, &
         dr, &
         taw, &
         et_ref, &
         et_adj, &
         t_early_sen, &
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
         beta &
         )

    call adjust_harvest_index( &
         hi_adj, &
         pre_adj, &
         f_pre, &
         f_pol, &
         f_post, &
         fpost_dwn, &
         fpost_upp, &
         s_cor1, &
         s_cor2, &
         yield_form, &
         hi_ref, &
         hi0, &
         dhi0, &
         b, &
         b_ns, &
         dhi_pre, &
         cc, &
         cc_min, &
         ksw_exp, &
         ksw_sto, &
         ksw_pol, &
         bio_temp_stress, &
         gdd, &
         gdd_up, &
         gdd_lo, &
         pol_heat_stress, &
         t_max, &
         t_max_up, &
         t_max_lo, &
         f_shp_b, &
         pol_cold_stress, &
         t_min, &
         t_min_up, &
         t_min_lo, &
         canopy_dev_end_cd, &
         hi_start_cd, &
         hi_end_cd, &
         yld_form_cd, &
         flowering_cd, &
         a_hi, &
         b_hi, &
         exc, &
         dap, &
         delayed_cds, &
         crop_type, &
         growing_season &
         )

    call update_crop_yield( &
         yield, &
         crop_mature, &
         maturity, &
         b, &
         hi_adj, &
         gdd_cum, &
         growing_season, &
         growing_season_day1, &
         calendar_type, &
         dap, &       
         delayed_cds, &
         delayed_gdds &
         )

    call update_root_zone_water( &
         thrz_act, &
         thrz_sat, &
         thrz_fc, &
         thrz_wilt, &
         thrz_dry, &
         thrz_aer, &
         taw, &
         dr, &
         th, &
         th_s, &
         th_fc, &
         th_wilt, &
         th_dry, &
         aer, &
         z_root, &
         z_min, &
         dz, &
         dz_sum, &
         layer_ix &
         )
    
  end subroutine update_aquacrop  
  
end module aquacrop

         
       

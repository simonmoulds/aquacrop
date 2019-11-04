module soil_evaporation_w
  use types
  use soil_evaporation, only: update_soil_evap
  implicit none

contains

  subroutine update_soil_evap_w( &
       prec, &
       et_ref, es_act, e_pot, &
       irr, irr_method, &
       infl, &
       th, th_sat, th_fc, th_wilt, th_dry, &
       surface_storage, &
       wet_surf, w_surf, w_stage_two, &
       cc, cc_adj, ccx_act, &
       evap_z, evap_z_min, evap_z_max, &
       rew, kex, ccxw, fwcc, f_evap, f_wrel_exp, &
       dz, dz_sum, &
       mulches, f_mulch, mulch_pct_gs, mulch_pct_os, &
       growing_season, senescence, premat_senes, &
       calendar_type, dap, delayed_cds, delayed_gdds, &
       time_step, evap_time_steps, &
       n_farm, n_crop, n_comp, n_cell)

    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_cell
    integer(int32), intent(in) :: calendar_type
    integer(int32), intent(in) :: time_step
    integer(int32), intent(in) :: evap_time_steps
    
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(in) :: th_sat, th_fc, th_wilt, th_dry
    real(real64), dimension(n_comp), intent(in) :: dz, dz_sum
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: prec, et_ref
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: infl
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: irr, wet_surf
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: rew
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: evap_z_min, evap_z_max    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc, cc_adj, ccx_act
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: kex, ccxw, fwcc, f_evap, f_wrel_exp
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_mulch, mulch_pct_gs, mulch_pct_os
    
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: irr_method
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: dap
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_cds
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_gdds
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: mulches
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: senescence
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: premat_senes
    
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: th
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: es_act, e_pot
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: surface_storage, w_surf, w_stage_two, evap_z

    integer(int32) :: i, j, k
    
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_soil_evap( &
                  prec(k,j,i), &
                  et_ref(k,j,i), es_act(k,j,i), e_pot(k,j,i), &
                  irr(k,j,i), irr_method(k,j,i), &
                  infl(k,j,i), &
                  th(k,:,j,i), th_sat(k,:,j,i), th_fc(k,:,j,i), th_wilt(k,:,j,i), th_dry(k,:,j,i), &
                  surface_storage(k,j,i), &
                  wet_surf(k,j,i), w_surf(k,j,i), w_stage_two(k,j,i), &
                  cc(k,j,i), cc_adj(k,j,i), ccx_act(k,j,i), &
                  evap_z(k,j,i), evap_z_min(k,j,i), evap_z_max(k,j,i), &
                  rew(k,j,i), kex(k,j,i), ccxw(k,j,i), fwcc(k,j,i), f_evap(k,j,i), f_wrel_exp(k,j,i), &
                  dz, dz_sum, &
                  mulches(k,j,i), f_mulch(k,j,i), mulch_pct_gs(k,j,i), mulch_pct_os(k,j,i), &
                  growing_season(k,j,i), senescence(k,j,i), premat_senes(k,j,i), &
                  calendar_type, dap(k,j,i), delayed_cds(k,j,i), delayed_gdds(k,j,i), &
                  time_step, evap_time_steps)
          end do
       end do
    end do
    
  end subroutine update_soil_evap_w
  
end module soil_evaporation_w

    

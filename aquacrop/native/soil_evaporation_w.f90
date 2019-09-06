module soil_evaporation_w
  use types
  use soil_evaporation, only: soil_evap
  implicit none

contains

  subroutine soil_evap_w( &
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
    
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(in) :: th_sat, th_fc, th_wilt, th_dry
    real(real64), dimension(n_comp), intent(in) :: dz, dz_sum
    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: prec, et_ref
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: infl
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: irr, wet_surf
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: rew
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: evap_z_min, evap_z_max    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc, cc_adj, ccx_act
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: kex, ccxw, fwcc, f_evap, f_wrel_exp
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_mulch, mulch_pct_gs, mulch_pct_os
    
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: irr_method
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: dap, delayed_cds, delayed_gdds
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: mulches
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: senescence, premat_senes
    
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: th
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: es_act, e_pot
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: surface_storage, w_surf, w_stage_two, evap_z

    integer(int32) :: i, j, k
    
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call soil_evap( &
                  prec(i,j,k), &
                  et_ref(i,j,k), es_act(i,j,k), e_pot(i,j,k), &
                  irr(i,j,k), irr_method(i,j,k), &
                  infl(i,j,k), &
                  th(i,j,:,k), th_sat(i,j,:,k), th_fc(i,j,:,k), th_wilt(i,j,:,k), th_dry(i,j,:,k), &
                  surface_storage(i,j,k), &
                  wet_surf(i,j,k), w_surf(i,j,k), w_stage_two(i,j,k), &
                  cc(i,j,k), cc_adj(i,j,k), ccx_act(i,j,k), &
                  evap_z(i,j,k), evap_z_min(i,j,k), evap_z_max(i,j,k), &
                  rew(i,j,k), kex(i,j,k), ccxw(i,j,k), fwcc(i,j,k), f_evap(i,j,k), f_wrel_exp(i,j,k), &
                  dz, dz_sum, &
                  mulches(i,j,k), f_mulch(i,j,k), mulch_pct_gs(i,j,k), mulch_pct_os(i,j,k), &
                  growing_season(i,j,k), senescence(i,j,k), premat_senes(i,j,k), &
                  calendar_type, dap(i,j,k), delayed_cds(i,j,k), delayed_gdds(i,j,k), &
                  time_step, evap_time_steps)
          end do
       end do
    end do
    
  end subroutine soil_evap_w
  
end module soil_evaporation_w

    

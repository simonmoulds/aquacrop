module transpiration_w
  use types
  use transpiration, only: update_transpiration, update_surf_trans, update_aeration_stress, adjust_surf_trans_stress

contains

  ! subroutine adjust_surf_trans_stress_w( &
  !      tr_pot, &
  !      ksa_aer, &
  !      aer_days, &
  !      ksw_stolin, &
  !      thrz_act, &
  !      thrz_sat, &
  !      thrz_aer, &
  !      lag_aer, &
  !      irr_method, &
  !      n_farm, n_crop, n_cell &
  !      )

  !   integer(int32), intent(in) :: n_farm, n_crop, n_cell
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_pot
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ksa_aer
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: aer_days
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ksw_stolin
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: thrz_act
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: thrz_sat
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: thrz_aer
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: lag_aer
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: irr_method
  !   integer(int32) :: i, j, k
  !   do i = 1, n_farm
  !      do j = 1, n_crop
  !         do k = 1, n_cell
  !            call adjust_surf_trans_stress( &
  !                 tr_pot(i,j,k), &
  !                 ksa_aer(i,j,k), &
  !                 aer_days(i,j,k), &
  !                 ksw_stolin(i,j,k), &
  !                 thrz_act(i,j,k), &
  !                 thrz_sat(i,j,k), &
  !                 thrz_aer(i,j,k), &
  !                 lag_aer(i,j,k), &
  !                 irr_method(i,j,k) &
  !                 )
  !         end do
  !      end do
  !   end do
  ! end subroutine adjust_surf_trans_stress_w  
             
  ! subroutine update_aeration_stress_w( &
  !      ksa_aer, &
  !      aer_days, &
  !      thrz_act, &
  !      thrz_sat, &
  !      thrz_aer, &
  !      lag_aer, &
  !      n_farm, n_crop, n_cell &
  !      )

  !   integer(int32), intent(in) :: n_farm, n_crop, n_cell
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ksa_aer
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: aer_days
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: thrz_act
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: thrz_sat
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: thrz_aer
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: lag_aer
  !   integer(int32) :: i, j, k
  !   do i = 1, n_farm
  !      do j = 1, n_crop
  !         do k = 1, n_cell
  !            call update_aeration_stress( &
  !                 ksa_aer(i,j,k), &
  !                 aer_days(i,j,k), &
  !                 thrz_act(i,j,k), &
  !                 thrz_sat(i,j,k), &
  !                 thrz_aer(i,j,k), &
  !                 lag_aer(i,j,k) &
  !                 )
  !         end do
  !      end do
  !   end do
  ! end subroutine update_aeration_stress_w
  
  ! subroutine update_surf_trans_w( &
  !      tr_pot0, &
  !      tr_pot_ns, &
  !      tr_pot, &
  !      tr_act0, &
  !      aer_days, &
  !      aer_days_comp, &
  !      age_days, &
  !      age_days_ns, &
  !      day_submrgd, &
  !      surface_storage, &
  !      cc, &
  !      et0, &
  !      max_canopy_cd, &
  !      kcb, &
  !      cc_adj, &
  !      cc_adj_ns, &
  !      ccxw, &
  !      ccxw_ns, &
  !      a_tr, &
  !      fage, &
  !      lag_aer, &
  !      co2_conc, &
  !      co2_refconc, &
  !      dap, &
  !      delayed_cds, &
  !      dz, &
  !      n_farm, n_crop, n_comp, n_cell &
  !      )

  !   integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_cell
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_pot0
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_pot_ns
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_pot
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_act0
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: aer_days
  !   integer(int32), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: aer_days_comp
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: age_days
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: age_days_ns
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: day_submrgd
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: surface_storage
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: et0
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: max_canopy_cd
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: kcb
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc_adj
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc_adj_ns
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ccxw
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ccxw_ns
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: a_tr
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: fage
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: lag_aer
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: co2_conc
  !   real(real64), intent(in) :: co2_refconc
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: dap
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_cds
  !   real(real64), dimension(n_comp), intent(in) :: dz
  !   integer(int32) :: i, j, k
    
  !   do i = 1, n_farm
  !      do j = 1, n_crop
  !         do k = 1, n_cell
  !            call update_surf_trans( &
  !                 tr_pot0(i,j,k), &
  !                 tr_pot_ns(i,j,k), &
  !                 tr_pot(i,j,k), &
  !                 tr_act0(i,j,k), &
  !                 aer_days(i,j,k), &
  !                 aer_days_comp(i,j,:,k), &
  !                 age_days(i,j,k), &
  !                 age_days_ns(i,j,k), &
  !                 day_submrgd(i,j,k), &
  !                 surface_storage(i,j,k), &
  !                 cc(i,j,k), &
  !                 et0(i,j,k), &
  !                 max_canopy_cd(i,j,k), &
  !                 kcb(i,j,k), &
  !                 cc_adj(i,j,k), &
  !                 cc_adj_ns(i,j,k), &
  !                 ccxw(i,j,k), &
  !                 ccxw_ns(i,j,k), &
  !                 a_tr(i,j,k), &
  !                 fage(i,j,k), &
  !                 lag_aer(i,j,k), &
  !                 co2_conc(i,j,k), &
  !                 co2_refconc, &
  !                 dap(i,j,k), &
  !                 delayed_cds(i,j,k), &
  !                 dz &
  !                 )
  !         end do
  !      end do
  !   end do
  ! end subroutine update_surf_trans_w
       
  subroutine update_transpiration_w( &
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
       surface_storage, &
       irr_net, &
       irr_net_cum, &
       cc, &
       et0, &
       th_sat, &
       th_fc, &
       th_wilt, &
       th_dry, &
       max_canopy_cd, &
       kcb, &
       ksw_stolin, &
       cc_adj, &
       cc_adj_ns, &
       cc_prev, &
       ccxw, &
       ccxw_ns, &
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
       growing_season, &
       n_farm, n_crop, n_comp, n_layer, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_pot0
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_pot_ns
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_act
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_act0
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: t_pot
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: tr_ratio    
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: aer_days
    integer(int32), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: aer_days_comp
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: th 
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_act
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_sat
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_fc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_wilt
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_dry
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: thrz_aer
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: taw
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: dr
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: age_days
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: age_days_ns
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: day_submrgd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: surface_storage
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: irr_net
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: irr_net_cum
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc
        
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: et0
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_sat
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_fc
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_wilt
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_dry
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: max_canopy_cd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: kcb
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ksw_stolin

    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc_adj
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc_adj_ns
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc_prev
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ccxw
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ccxw_ns

    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_root
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: r_cor
    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_min
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: a_tr
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: aer
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: fage
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: lag_aer
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: sx_bot
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: sx_top
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: et_adj
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo2
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up2
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w2

    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: irr_method
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: net_irr_smt
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: co2_conc
    real(real64), intent(in) :: co2_refconc
    
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: dap
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_cds
    real(real64), dimension(n_comp), intent(in) :: dz
    real(real64), dimension(n_comp), intent(in) :: dz_sum
    integer(int32), dimension(n_comp), intent(in) :: layer_ix
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season

    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_transpiration( &
                  tr_pot0(i,j,k), &
                  tr_pot_ns(i,j,k), &
                  tr_act(i,j,k), &
                  tr_act0(i,j,k), &
                  t_pot(i,j,k), &
                  tr_ratio(i,j,k), &
                  aer_days(i,j,k), &
                  aer_days_comp(i,j,:,k), &
                  th(i,j,:,k), &
                  thrz_act(i,j,k), &
                  thrz_sat(i,j,k), &
                  thrz_fc(i,j,k), &
                  thrz_wilt(i,j,k), &
                  thrz_dry(i,j,k), &
                  thrz_aer(i,j,k), &
                  taw(i,j,k), &
                  dr(i,j,k), &
                  age_days(i,j,k), &
                  age_days_ns(i,j,k), &
                  day_submrgd(i,j,k), &
                  surface_storage(i,j,k), &
                  irr_net(i,j,k), &
                  irr_net_cum(i,j,k), &
                  cc(i,j,k), &
                  et0(i,j,k), &
                  th_sat(i,j,:,k), &
                  th_fc(i,j,:,k), &
                  th_wilt(i,j,:,k), &
                  th_dry(i,j,:,k), &
                  max_canopy_cd(i,j,k), &
                  kcb(i,j,k), &
                  ksw_stolin(i,j,k), &
                  cc_adj(i,j,k), &
                  cc_adj_ns(i,j,k), &
                  cc_prev(i,j,k), &
                  ccxw(i,j,k), &
                  ccxw_ns(i,j,k), &
                  z_root(i,j,k), &
                  r_cor(i,j,k), &
                  z_min(i,j,k), &
                  a_tr(i,j,k), &
                  aer(i,j,k), &
                  fage(i,j,k), &
                  lag_aer(i,j,k), &
                  sx_bot(i,j,k), &
                  sx_top(i,j,k), &
                  et_adj(i,j,k), &
                  p_lo2(i,j,k), &
                  p_up2(i,j,k), &
                  f_shape_w2(i,j,k), &
                  irr_method(i,j,k), &
                  net_irr_smt(i,j,k), &
                  co2_conc(i,j,k), &
                  co2_refconc, &
                  dap(i,j,k), &
                  delayed_cds(i,j,k), &
                  dz, &
                  dz_sum, &
                  layer_ix, &
                  growing_season(i,j,k) &
                  )
          end do
       end do
    end do
  end subroutine update_transpiration_w
end module transpiration_w

             
             

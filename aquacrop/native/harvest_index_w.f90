module harvest_index_w
  use types
  use harvest_index, only: update_harvest_index, adjust_harvest_index, hi_adj_pre_anthesis, hi_adj_pol, hi_adj_post_anthesis
  implicit none

contains
    
  subroutine update_harvest_index_w( &
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
       growing_season, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: hi_ref
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: pct_lag_phase
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: yield_form
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc_prev
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc_min
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ccx
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_ini
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi0
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: higc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start_cd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_lin_switch
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: dhi_linear
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: gdd_cum
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: dap
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_cds
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_gdds
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: crop_type
    integer(int32), intent(in) :: calendar_type
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season
    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_harvest_index( &
                  hi_ref(i,j,k), &
                  pct_lag_phase(i,j,k), &
                  yield_form(i,j,k), &
                  cc_prev(i,j,k), &
                  cc_min(i,j,k), &
                  ccx(i,j,k), &
                  hi_ini(i,j,k), &
                  hi0(i,j,k), &
                  higc(i,j,k), &
                  hi_start(i,j,k), &
                  hi_start_cd(i,j,k), &
                  t_lin_switch(i,j,k), &
                  dhi_linear(i,j,k), &
                  gdd_cum(i,j,k), &
                  dap(i,j,k), &
                  delayed_cds(i,j,k), &
                  delayed_gdds(i,j,k), &
                  crop_type(i,j,k), &
                  calendar_type, &
                  growing_season(i,j,k) &
                  )
          end do
       end do
    end do
    
  end subroutine update_harvest_index_w

  subroutine hi_adj_pre_anthesis_w( &
       f_pre, &
       b, &
       b_ns, &
       dhi_pre, &
       cc, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: f_pre
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: b
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: b_ns
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: dhi_pre
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call hi_adj_pre_anthesis( &
                  f_pre(i,j,k), &
                  b(i,j,k), &
                  b_ns(i,j,k), &
                  dhi_pre(i,j,k), &
                  cc(i,j,k) &
                  )
          end do
       end do
    end do
  end subroutine hi_adj_pre_anthesis_w  

  subroutine hi_adj_pol_w( &
       f_pol, &
       hi_t, &
       flowering_cd, &
       cc, &
       cc_min, &
       ksw_pol, &
       kst_polc, &
       kst_polh, &
       exc, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: f_pol
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_t
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: flowering_cd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc_min
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ksw_pol
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: kst_polc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: kst_polh
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: exc
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call hi_adj_pol( &
                  f_pol(i,j,k), &
                  hi_t(i,j,k), &
                  flowering_cd(i,j,k), &
                  cc(i,j,k), &
                  cc_min(i,j,k), &
                  ksw_pol(i,j,k), &
                  kst_polc(i,j,k), &
                  kst_polh(i,j,k), &
                  exc(i,j,k) &
                  )
          end do
       end do
    end do
  end subroutine hi_adj_pol_w

  subroutine hi_adj_post_anthesis_w( &
       f_post, &
       fpost_dwn, &
       fpost_upp, &
       s_cor1, &
       s_cor2, &
       f_pre, &
       cc, &
       ksw_exp, &
       ksw_sto, &
       canopy_dev_end_cd, &
       hi_start_cd, &
       hi_end_cd, &
       yld_form_cd, &
       a_hi, &
       b_hi, &
       dap, &
       delayed_cds, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: f_post
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: fpost_dwn
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: fpost_upp
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: s_cor1
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: s_cor2
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_pre
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ksw_exp
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ksw_sto
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: canopy_dev_end_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_end_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: yld_form_cd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: a_hi
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: b_hi
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: dap
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_cds
    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call hi_adj_post_anthesis( &
                  f_post(i,j,k), &
                  fpost_dwn(i,j,k), &
                  fpost_upp(i,j,k), &
                  s_cor1(i,j,k), &
                  s_cor2(i,j,k), &
                  f_pre(i,j,k), &
                  cc(i,j,k), &
                  ksw_exp(i,j,k), &
                  ksw_sto(i,j,k), &
                  canopy_dev_end_cd(i,j,k), &
                  hi_start_cd(i,j,k), &
                  hi_end_cd(i,j,k), &
                  yld_form_cd(i,j,k), &
                  a_hi(i,j,k), &
                  b_hi(i,j,k), &
                  dap(i,j,k), &
                  delayed_cds(i,j,k) &
                  )
          end do
       end do
    end do
  end subroutine hi_adj_post_anthesis_w
  
  subroutine adjust_harvest_index_w( &
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
       kst_polc, &
       kst_polh, &
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
       growing_season, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: hi_adj
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: pre_adj
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: f_pre
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: f_pol
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: f_post
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: fpost_dwn
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: fpost_upp
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: s_cor1
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: s_cor2
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: yield_form
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_ref
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi0
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: dhi0
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: b
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: b_ns
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: dhi_pre
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc_min
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ksw_exp
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ksw_sto
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ksw_pol
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: kst_polc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: kst_polh
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: canopy_dev_end_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_end_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: yld_form_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: flowering_cd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: a_hi
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: b_hi
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: exc
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: dap
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_cds
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: crop_type
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season
    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call adjust_harvest_index( &
                  hi_adj(i,j,k), &
                  pre_adj(i,j,k), &
                  f_pre(i,j,k), &
                  f_pol(i,j,k), &
                  f_post(i,j,k), &
                  fpost_dwn(i,j,k), &
                  fpost_upp(i,j,k), &
                  s_cor1(i,j,k), &
                  s_cor2(i,j,k), &
                  yield_form(i,j,k), &
                  hi_ref(i,j,k), &
                  hi0(i,j,k), &
                  dhi0(i,j,k), &
                  b(i,j,k), &
                  b_ns(i,j,k), &
                  dhi_pre(i,j,k), &
                  cc(i,j,k), &
                  cc_min(i,j,k), &
                  ksw_exp(i,j,k), &
                  ksw_sto(i,j,k), &
                  ksw_pol(i,j,k), &
                  kst_polc(i,j,k), &
                  kst_polh(i,j,k), &
                  canopy_dev_end_cd(i,j,k), &
                  hi_start_cd(i,j,k), &
                  hi_end_cd(i,j,k), &
                  yld_form_cd(i,j,k), &
                  flowering_cd(i,j,k), &
                  a_hi(i,j,k), &
                  b_hi(i,j,k), &
                  exc(i,j,k), &
                  dap(i,j,k), &
                  delayed_cds(i,j,k), &
                  crop_type(i,j,k), &
                  growing_season(i,j,k) &
                  )
          end do
       end do
    end do
  end subroutine adjust_harvest_index_w  
                  
end module harvest_index_w

             
                  

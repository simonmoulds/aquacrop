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
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: hi_ref
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: pct_lag_phase
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: yield_form
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc_prev
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc_min
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: ccx
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_ini
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi0
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: higc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_start
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_start_cd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: t_lin_switch
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: dhi_linear
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: gdd_cum
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: dap
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_cds
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_gdds
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: crop_type
    integer(int32), intent(in) :: calendar_type
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season
    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_harvest_index( &
                  hi_ref(k,j,i), &
                  pct_lag_phase(k,j,i), &
                  yield_form(k,j,i), &
                  cc_prev(k,j,i), &
                  cc_min(k,j,i), &
                  ccx(k,j,i), &
                  hi_ini(k,j,i), &
                  hi0(k,j,i), &
                  higc(k,j,i), &
                  hi_start(k,j,i), &
                  hi_start_cd(k,j,i), &
                  t_lin_switch(k,j,i), &
                  dhi_linear(k,j,i), &
                  gdd_cum(k,j,i), &
                  dap(k,j,i), &
                  delayed_cds(k,j,i), &
                  delayed_gdds(k,j,i), &
                  crop_type(k,j,i), &
                  calendar_type, &
                  growing_season(k,j,i) &
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
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_pre
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: b
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: b_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: dhi_pre
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call hi_adj_pre_anthesis( &
                  f_pre(k,j,i), &
                  b(k,j,i), &
                  b_ns(k,j,i), &
                  dhi_pre(k,j,i), &
                  cc(k,j,i) &
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
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_pol
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_t
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: flowering_cd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc_min
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: ksw_pol
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: kst_polc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: kst_polh
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: exc
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call hi_adj_pol( &
                  f_pol(k,j,i), &
                  hi_t(k,j,i), &
                  flowering_cd(k,j,i), &
                  cc(k,j,i), &
                  cc_min(k,j,i), &
                  ksw_pol(k,j,i), &
                  kst_polc(k,j,i), &
                  kst_polh(k,j,i), &
                  exc(k,j,i) &
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
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_post
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: fpost_dwn
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: fpost_upp
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: s_cor1
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: s_cor2
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_pre
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: ksw_exp
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: ksw_sto
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: canopy_dev_end_cd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_start_cd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_end_cd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: yld_form_cd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: a_hi
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: b_hi
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: dap
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_cds
    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call hi_adj_post_anthesis( &
                  f_post(k,j,i), &
                  fpost_dwn(k,j,i), &
                  fpost_upp(k,j,i), &
                  s_cor1(k,j,i), &
                  s_cor2(k,j,i), &
                  f_pre(k,j,i), &
                  cc(k,j,i), &
                  ksw_exp(k,j,i), &
                  ksw_sto(k,j,i), &
                  canopy_dev_end_cd(k,j,i), &
                  hi_start_cd(k,j,i), &
                  hi_end_cd(k,j,i), &
                  yld_form_cd(k,j,i), &
                  a_hi(k,j,i), &
                  b_hi(k,j,i), &
                  dap(k,j,i), &
                  delayed_cds(k,j,i) &
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
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: hi_adj
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: pre_adj
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_pre
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_pol
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: f_post
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: fpost_dwn
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: fpost_upp
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: s_cor1
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: s_cor2
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: yield_form
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_ref
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi0
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: dhi0
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: b
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: b_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: dhi_pre
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc_min
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: ksw_exp
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: ksw_sto
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: ksw_pol
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: kst_polc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: kst_polh
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: canopy_dev_end_cd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_start_cd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_end_cd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: yld_form_cd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: flowering_cd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: a_hi
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: b_hi
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: exc
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: dap
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_cds
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: crop_type
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season
    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call adjust_harvest_index( &
                  hi_adj(k,j,i), &
                  pre_adj(k,j,i), &
                  f_pre(k,j,i), &
                  f_pol(k,j,i), &
                  f_post(k,j,i), &
                  fpost_dwn(k,j,i), &
                  fpost_upp(k,j,i), &
                  s_cor1(k,j,i), &
                  s_cor2(k,j,i), &
                  yield_form(k,j,i), &
                  hi_ref(k,j,i), &
                  hi0(k,j,i), &
                  dhi0(k,j,i), &
                  b(k,j,i), &
                  b_ns(k,j,i), &
                  dhi_pre(k,j,i), &
                  cc(k,j,i), &
                  cc_min(k,j,i), &
                  ksw_exp(k,j,i), &
                  ksw_sto(k,j,i), &
                  ksw_pol(k,j,i), &
                  kst_polc(k,j,i), &
                  kst_polh(k,j,i), &
                  canopy_dev_end_cd(k,j,i), &
                  hi_start_cd(k,j,i), &
                  hi_end_cd(k,j,i), &
                  yld_form_cd(k,j,i), &
                  flowering_cd(k,j,i), &
                  a_hi(k,j,i), &
                  b_hi(k,j,i), &
                  exc(k,j,i), &
                  dap(k,j,i), &
                  delayed_cds(k,j,i), &
                  crop_type(k,j,i), &
                  growing_season(k,j,i) &
                  )
          end do
       end do
    end do
  end subroutine adjust_harvest_index_w  
                  
end module harvest_index_w

             
                  

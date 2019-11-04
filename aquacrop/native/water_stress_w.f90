module water_stress_w
  use types
  use water_stress, only: update_water_stress
  implicit none
  
contains

  subroutine update_water_stress_w( &
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
       beta, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ksw_exp
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ksw_sto
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ksw_sen
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ksw_pol
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ksw_stolin
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: dr
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: taw
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: et_ref
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: et_adj
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: t_early_sen
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
    integer(int32), intent(in) :: beta
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_water_stress( &
                  ksw_exp(k,j,i), &
                  ksw_sto(k,j,i), &
                  ksw_sen(k,j,i), &
                  ksw_pol(k,j,i), &
                  ksw_stolin(k,j,i), &
                  dr(k,j,i), &
                  taw(k,j,i), &
                  et_ref(k,j,i), &
                  et_adj(k,j,i), &
                  t_early_sen(k,j,i), &
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
                  beta &
                  )
          end do
       end do
    end do
    
  end subroutine update_water_stress_w
  
end module water_stress_w


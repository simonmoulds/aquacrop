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
    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ksw_exp
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ksw_sto
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ksw_sen
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ksw_pol
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ksw_stolin
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: dr
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: taw
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: et_ref
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: et_adj
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_early_sen
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up1
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up2
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up3
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up4
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo1
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo2
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo3
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo4
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w1
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w2
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w3
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w4
    integer(int32), intent(in) :: beta

    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_water_stress( &
                  ksw_exp(i,j,k), &
                  ksw_sto(i,j,k), &
                  ksw_sen(i,j,k), &
                  ksw_pol(i,j,k), &
                  ksw_stolin(i,j,k), &
                  dr(i,j,k), &
                  taw(i,j,k), &
                  et_ref(i,j,k), &
                  et_adj(i,j,k), &
                  t_early_sen(i,j,k), &
                  p_up1(i,j,k), &
                  p_up2(i,j,k), &
                  p_up3(i,j,k), &
                  p_up4(i,j,k), &
                  p_lo1(i,j,k), &
                  p_lo2(i,j,k), &
                  p_lo3(i,j,k), &
                  p_lo4(i,j,k), &
                  f_shape_w1(i,j,k), &
                  f_shape_w2(i,j,k), &
                  f_shape_w3(i,j,k), &
                  f_shape_w4(i,j,k), &
                  beta &
                  )
          end do
       end do
    end do
    
  end subroutine update_water_stress_w
  
end module water_stress_w


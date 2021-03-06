module pre_irr_w
  use types
  use pre_irr, only: update_pre_irr
  implicit none

contains

  subroutine update_pre_irr_w( &
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
       layer_ix, &
       n_farm, n_crop, n_comp, n_layer, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: pre_irr
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: th
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: irr_method
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: dap
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_root
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_min
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: net_irr_smt
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_fc
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_wilt
    real(real64), dimension(n_comp), intent(in) :: dz
    real(real64), dimension(n_comp), intent(in) :: dz_sum
    integer(int32), dimension(n_comp), intent(in) :: layer_ix
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_pre_irr( &
                  pre_irr(k,j,i), &
                  th(k,:,j,i), &
                  irr_method(k,j,i), &
                  dap(k,j,i), &
                  z_root(k,j,i), &
                  z_min(k,j,i), &
                  net_irr_smt(k,j,i), &
                  th_fc(k,:,j,i), &
                  th_wilt(k,:,j,i), &
                  dz, &
                  dz_sum, &
                  layer_ix &
                  )
          end do
       end do
    end do
    
  end subroutine update_pre_irr_w
  
end module pre_irr_w

                  
    
    
  
  

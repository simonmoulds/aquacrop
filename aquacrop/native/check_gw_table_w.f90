module check_gw_table_w
  use types
  use check_gw_table, only: update_check_gw_table
  implicit none

contains

  subroutine update_check_gw_table_w( &
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
       n_farm, n_crop, n_comp, n_layer, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
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

    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_check_gw_table( &
                  th(k,:,j,i), &
                  th_fc_adj(k,:,j,i), &
                  wt_in_soil(k,j,i), &       
                  th_s(k,:,j,i), &
                  th_fc(k,:,j,i), &
                  wt, &
                  variable_wt, &
                  zgw(k), &
                  dz, &
                  layer_ix &
                  )
          end do
       end do
    end do
    
  end subroutine update_check_gw_table_w
  
end module check_gw_table_w

                  
       

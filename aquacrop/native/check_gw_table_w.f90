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
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: th
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: th_fc_adj
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: wt_in_soil
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_s
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_fc    
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
                  th(i,j,:,k), &
                  th_fc_adj(i,j,:,k), &
                  wt_in_soil(i,j,k), &       
                  th_s(i,j,:,k), &
                  th_fc(i,j,:,k), &
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

                  
       

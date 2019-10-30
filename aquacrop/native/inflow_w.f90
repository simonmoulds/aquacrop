module inflow_w
  use types
  use inflow, only: update_inflow
  implicit none

contains
  
  subroutine update_inflow_w(&
       gw_in, &
       th, &
       water_table, &
       z_gw, &
       th_sat, &
       dz, &
       layer_ix, &
       n_farm, n_crop, n_comp, n_layer, n_cell &
       )

    integer(int32) :: n_farm, n_crop, n_comp, n_layer, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: gw_in
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: th
    integer(int32), intent(in) :: water_table
    real(real64), dimension(n_cell), intent(in) :: z_gw
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_sat
    real(real64), dimension(n_comp), intent(in) :: dz
    real(real64), dimension(n_comp), intent(in) :: layer_ix
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_inflow( &
                  gw_in(i,j,k), &
                  th(i,j,:,k), &
                  water_table, &
                  z_gw(k), &
                  th_sat(i,j,:,k), &
                  dz, &
                  layer_ix &
                  )
          end do
       end do
    end do
    
  end subroutine update_inflow_w
  
end module inflow_w

             
             
       

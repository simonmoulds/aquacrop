module infiltration_w
  use types
  use infiltration, only: update_infl
  implicit none
  
contains
  
  subroutine update_infl_w( &
       infl, &
       surf_stor, &
       flux_out, &
       deep_perc, &
       runoff, &
       th, &
       irr, &
       app_eff, &
       bund, &
       z_bund, &
       th_sat, &
       th_fc, &
       th_fc_adj, &       
       k_sat, &
       tau, &
       dz, &
       layer_ix, &
       n_farm, n_crop, n_comp, n_layer, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: infl 
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: surf_stor
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: flux_out
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: deep_perc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: runoff
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: th
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: irr
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: app_eff
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: bund
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_bund
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_sat
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_fc
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(in) :: th_fc_adj
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: k_sat
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: tau
    real(real64), dimension(n_comp), intent(in) :: dz
    integer(int32), dimension(n_comp), intent(in) :: layer_ix
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_infl( &
                  infl(i,j,k), &
                  surf_stor(i,j,k), &
                  flux_out(i,j,:,k), &
                  deep_perc(i,j,k), &
                  runoff(i,j,k), &
                  th(i,j,:,k), &
                  irr(i,j,k), &
                  app_eff(i,j,k), &
                  bund(i,j,k), &
                  z_bund(i,j,k), &
                  th_sat(i,j,:,k), &
                  th_fc(i,j,:,k), &
                  th_fc_adj(i,j,:,k), &       
                  k_sat(i,j,:,k), &
                  tau(i,j,:,k), &
                  dz, &
                  layer_ix &
                  )
          end do
       end do
    end do
    
  end subroutine update_infl_w
  
end module infiltration_w

    

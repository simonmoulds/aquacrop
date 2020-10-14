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
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: infl 
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: surf_stor
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: flux_out
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: deep_perc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: runoff
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: th
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: irr
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: app_eff
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: bund
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_bund
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_sat
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_fc
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(in) :: th_fc_adj
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: k_sat
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: tau
    real(real64), dimension(n_comp), intent(in) :: dz
    integer(int32), dimension(n_comp), intent(in) :: layer_ix
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_infl( &
                  infl(k,j,i), &
                  surf_stor(k,j,i), &
                  flux_out(k,:,j,i), &
                  deep_perc(k,j,i), &
                  runoff(k,j,i), &
                  th(k,:,j,i), &
                  irr(k,j,i), &
                  app_eff(k,j,i), &
                  bund(k,j,i), &
                  z_bund(k,j,i), &
                  th_sat(k,:,j,i), &
                  th_fc(k,:,j,i), &
                  th_fc_adj(k,:,j,i), &       
                  k_sat(k,:,j,i), &
                  tau(k,:,j,i), &
                  dz, &
                  layer_ix &
                  )
          end do
       end do
    end do
    
  end subroutine update_infl_w
  
end module infiltration_w

    

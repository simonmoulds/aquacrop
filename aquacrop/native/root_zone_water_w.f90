module root_zone_water_w
  use types
  use root_zone_water, only: update_root_zone_water
  implicit none
  
contains

  subroutine update_root_zone_water_w( &
       thrz_act, &
       thrz_sat, &
       thrz_fc, &
       thrz_wilt, &
       thrz_dry, &
       thrz_aer, &
       taw, &
       dr, &
       th, &
       th_sat, &
       th_fc, &
       th_wilt, &
       th_dry, &
       aer, &
       z_root, &
       z_min, &
       dz, &
       dz_sum, &
       layer_ix, &
       n_farm, n_crop, n_comp, n_layer, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_act
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_sat
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_fc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_wilt
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_dry
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: thrz_aer
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: taw
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: dr
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(in) :: th
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_sat
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_fc
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_wilt
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_dry
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: aer
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_root
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_min
    real(real64), dimension(n_comp), intent(in) :: dz
    real(real64), dimension(n_comp), intent(in) :: dz_sum
    integer(int32), dimension(n_comp), intent(in) :: layer_ix
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_root_zone_water( &
                  thrz_act(k,j,i), &
                  thrz_sat(k,j,i), &
                  thrz_fc(k,j,i), &
                  thrz_wilt(k,j,i), &
                  thrz_dry(k,j,i), &
                  thrz_aer(k,j,i), &
                  taw(k,j,i), &
                  dr(k,j,i), &
                  th(k,:,j,i), &
                  th_sat(k,:,j,i), &
                  th_fc(k,:,j,i), &
                  th_wilt(k,:,j,i), &
                  th_dry(k,:,j,i), &
                  aer(k,j,i), &
                  z_root(k,j,i), &
                  z_min(k,j,i), &
                  dz, &
                  dz_sum, &
                  layer_ix &
                  )
          end do
       end do
    end do
    
  end subroutine update_root_zone_water_w
  
end module root_zone_water_w

             
                  
    

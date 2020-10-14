module germination_w
  use types
  use germination, only: update_germ
  implicit none

contains

  subroutine update_germ_w( &
       germ, &
       delayed_cds, &
       delayed_gdds, &
       gdd, &
       th, &
       th_fc, &
       th_wilt, &
       z_germ, &
       germ_thr, &
       dz, &
       dz_sum, &
       layer_ix, &
       growing_season, &
       n_farm, n_crop, n_comp, n_layer, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: germ
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: delayed_cds
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: delayed_gdds
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: gdd
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(in) :: th
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_fc
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_wilt
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_germ
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: germ_thr
    real(real64), dimension(n_comp), intent(in) :: dz
    real(real64), dimension(n_comp), intent(in) :: dz_sum
    integer(int32), dimension(n_comp), intent(in) :: layer_ix
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season

    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_germ( &
                  germ(k,j,i), &
                  delayed_cds(k,j,i), &
                  delayed_gdds(k,j,i), &
                  gdd(k,j,i), &
                  th(k,:,j,i), &
                  th_fc(k,:,j,i), &
                  th_wilt(k,:,j,i), &
                  z_germ(k,j,i), &
                  germ_thr(k,j,i), &
                  dz, &
                  dz_sum, &
                  layer_ix, &
                  growing_season(k,j,i) &
                  )
          end do
       end do
    end do

  end subroutine update_germ_w
  
  ! subroutine update_germ_w( &
  !      germ, &
  !      delayed_cds, &
  !      delayed_gdds, &
  !      th, &
  !      th_fc, &
  !      th_wilt, &
  !      z_germ, &
  !      germ_thr, &
  !      dz, &
  !      dz_sum, &
  !      layer_ix, &
  !      growing_season, &
  !      n_farm, n_crop, n_comp, n_layer, n_cell &
  !      )

  !   integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: germ
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: delayed_cds
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: delayed_gdds
  !   real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(in) :: th
  !   real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_fc
  !   real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_wilt
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_germ
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: germ_thr
  !   real(real64), dimension(n_comp), intent(in) :: dz
  !   real(real64), dimension(n_comp), intent(in) :: dz_sum
  !   integer(int32), dimension(n_comp), intent(in) :: layer_ix
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season

  !   integer(int32) :: i, j, k
  !   do i = 1, n_farm
  !      do j = 1, n_crop
  !         do k = 1, n_cell
  !            call update_germ( &
  !                 germ(i,j,k), &
  !                 delayed_cds(i,j,k), &
  !                 delayed_gdds(i,j,k), &
  !                 th(i,j,:,k), &
  !                 th_fc(i,j,:,k), &
  !                 th_wilt(i,j,:,k), &
  !                 z_germ(i,j,k), &
  !                 germ_thr(i,j,k), &
  !                 dz, &
  !                 dz_sum, &
  !                 layer_ix, &
  !                 growing_season(i,j,k) &
  !                 )
  !         end do
  !      end do
  !   end do
    
  ! end subroutine update_germ_w
  
end module germination_w

                  
    
       

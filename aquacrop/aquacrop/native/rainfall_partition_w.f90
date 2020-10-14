module rainfall_partition_w
  use types
  use rainfall_partition, only: update_rain_part
  implicit none

contains
  
  subroutine update_rain_part_w( &
       runoff, &
       infl, &
       prec, &
       th, &
       days_submrgd, &
       bund, &
       z_bund, &
       th_fc, &
       th_wilt, &
       cn, &
       adj_cn, &
       z_cn, &
       cn_bot, &
       cn_top, &
       dz, &
       dz_sum, &
       layer_ix, &
       n_farm, n_crop, n_comp, n_layer, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: runoff 
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: infl
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: prec
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(in) :: th
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: days_submrgd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: bund
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_bund
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_fc
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_wilt    
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: cn
    integer(int32), intent(in) :: adj_cn
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_cn
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cn_bot
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cn_top
    real(real64), dimension(n_comp), intent(in) :: dz
    real(real64), dimension(n_comp), intent(in) :: dz_sum
    integer(int32), dimension(n_comp), intent(in) :: layer_ix
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_rain_part( &
                  runoff(k,j,i), &
                  infl(k,j,i), &
                  prec(k,j,i), &
                  th(k,:,j,i), &
                  days_submrgd(k,j,i), &
                  bund(k,j,i), &
                  z_bund(k,j,i), &
                  th_fc(k,:,j,i), &
                  th_wilt(k,:,j,i), &
                  cn(k,j,i), &
                  adj_cn, &
                  z_cn(k,j,i), &
                  cn_bot(k,j,i), &
                  cn_top(k,j,i), &
                  dz, &
                  dz_sum, &
                  layer_ix &
                  )
          end do
       end do
    end do
    
  end subroutine update_rain_part_w
  
end module rainfall_partition_w

                  

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
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: runoff 
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: infl
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: prec
    real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(in) :: th
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: days_submrgd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: bund
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_bund
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_fc
    real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_wilt    
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: cn
    integer(int32), intent(in) :: adj_cn
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_cn
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cn_bot
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cn_top
    real(real64), dimension(n_comp), intent(in) :: dz
    real(real64), dimension(n_comp), intent(in) :: dz_sum
    real(real64), dimension(n_comp), intent(in) :: layer_ix
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_rain_part( &
                  runoff(i,j,k), &
                  infl(i,j,k), &
                  prec(i,j,k), &
                  th(i,j,:,k), &
                  days_submrgd(i,j,k), &
                  bund(i,j,k), &
                  z_bund(i,j,k), &
                  th_fc(i,j,:,k), &
                  th_wilt(i,j,:,k), &
                  cn(i,j,k), &
                  adj_cn, &
                  z_cn(i,j,k), &
                  cn_bot(i,j,k), &
                  cn_top(i,j,k), &
                  dz, &
                  dz_sum, &
                  layer_ix &
                  )
          end do
       end do
    end do
    
  end subroutine update_rain_part_w
  
end module rainfall_partition_w

                  

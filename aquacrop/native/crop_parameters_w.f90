module crop_parameters_w
  use types
  use crop_parameters
  implicit none

contains

  subroutine adjust_pd_hd_w( &
       planting_date_adj, &
       harvest_date_adj, &
       planting_date, &
       harvest_date, &
       day_of_year, &
       time_step, &
       leap_year, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: planting_date_adj
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: harvest_date_adj
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: planting_date
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: harvest_date
    integer(int32), intent(in) :: day_of_year
    integer(int32), intent(in) :: time_step
    integer(int32), intent(in) :: leap_year
    integer(int32) :: i, j, k
    
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call adjust_pd_hd( &
                  planting_date_adj(k,j,i), &
                  harvest_date_adj(k,j,i), &
                  planting_date(k,j,i), &
                  harvest_date(k,j,i), &
                  day_of_year, &
                  time_step, &
                  leap_year &
                  )
          end do
       end do
    end do
    
  end subroutine adjust_pd_hd_w

  subroutine update_growing_season_w( &
       growing_season_index, &
       growing_season_day_one, &
       dap, &
       pd, &
       hd, &
       crop_dead, &
       crop_mature, &
       doy, &
       time_step, &
       year_start_num, &
       end_time_num, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: growing_season_index
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: growing_season_day_one
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: dap
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: pd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: hd
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: crop_dead
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: crop_mature
    integer(int32), intent(in) :: doy
    integer(int32), intent(in) :: time_step
    integer(int32), intent(in) :: year_start_num
    integer(int32), intent(in) :: end_time_num
    integer(int32) :: i, j, k
    
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_growing_season( &
                  growing_season_index(k,j,i), &
                  growing_season_day_one(k,j,i), &
                  dap(k,j,i), &
                  pd(k,j,i), &
                  hd(k,j,i), &
                  crop_dead(k,j,i), &
                  crop_mature(k,j,i), &
                  doy, &
                  time_step, &
                  year_start_num, &
                  end_time_num &
                  )
          end do
       end do
    end do
  end subroutine update_growing_season_w                    
    
end module crop_parameters_w

  
    
  

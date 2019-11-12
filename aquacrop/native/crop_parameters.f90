module crop_parameters
  use types
  implicit none

  real(real64), parameter :: zero = 0.0d0
  
contains

  subroutine adjust_pd_hd( &
       planting_date_adj, &
       harvest_date_adj, &
       planting_date, &
       harvest_date, &
       day_of_year, &
       time_step, &
       leap_year &
       )

    integer(int32), intent(inout) :: planting_date_adj
    integer(int32), intent(inout) :: harvest_date_adj
    integer(int32), intent(in) :: planting_date
    integer(int32), intent(in) :: harvest_date
    integer(int32), intent(in) :: day_of_year
    integer(int32), intent(in) :: time_step
    integer(int32), intent(in) :: leap_year

    if ( time_step == 1 .or. day_of_year == 1 ) then
       if ( leap_year == 1 ) then
          if ( planting_date >= 60 ) then
             planting_date_adj = planting_date + 1
          else
             planting_date_adj = planting_date
          end if

          if ( harvest_date >= 60 .and. harvest_date > planting_date_adj ) then
             harvest_date_adj = harvest_date + 1
          else
             harvest_date_adj = harvest_date
          end if
       end if
    end if    
    
  end subroutine adjust_pd_hd

  subroutine update_growing_season( &
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
       end_time_num &
       )

    integer(int32), intent(inout) :: growing_season_index
    integer(int32), intent(inout) :: growing_season_day_one
    integer(int32), intent(inout) :: dap
    integer(int32), intent(in) :: pd
    integer(int32), intent(in) :: hd
    integer(int32), intent(in) :: crop_dead
    integer(int32), intent(in) :: crop_mature
    integer(int32), intent(in) :: doy
    integer(int32), intent(in) :: time_step
    integer(int32), intent(in) :: year_start_num
    integer(int32), intent(in) :: end_time_num
    integer(int32) :: hd_num
    logical :: cond1, cond2, cond3, cond4, cond5
    
    if ( doy == pd ) then
       growing_season_day_one = 1
       dap = 0
    else
       growing_season_day_one = 0
    end if
    
    hd_num = year_start_num + (hd - 1)
    cond1 = pd < hd .and. ( pd <= doy .and. doy <= hd )
    cond2 = pd > hd .and. ( pd <= doy .or. doy <= hd )
    cond3 = hd_num <= end_time_num
    cond4 = (doy - pd) < time_step
    cond5 = (crop_dead == 0 .and. crop_mature == 0)
    if ( (cond1 .or. cond2) .and. cond3 .and. cond4 .and. cond5 ) then
       growing_season_index = 1
       dap = dap + 1
    else
       growing_season_index = 0
       dap = 0
    end if
    
  end subroutine update_growing_season
  
end module crop_parameters

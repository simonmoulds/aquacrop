module crop_parameters
  use types
  implicit none

  real(real64), parameter :: zero = 0.0d0
  
contains

  subroutine compute_max_canopy( &
       max_canopy, &
       emergence, &
       ccx, &
       cc0, &
       cgc &
       )
    real(real64), intent(inout) :: max_canopy
    real(real64), intent(in) :: emergence
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: cgc
    real(real64) :: a
    a = (0.25 * ccx * ccx / cc0) / (ccx - (0.98 * ccx))
    if ( a .eq. 0. ) then
       max_canopy = nint(emergence) ! NOT SURE ABOUT THIS
    else       
       max_canopy = nint(emergence + log((0.25 * ccx * ccx / cc0) / (ccx - (0.98 * ccx))) / cgc)
    end if        
  end subroutine compute_max_canopy

  subroutine compute_hi_end( &
       hi_end, &
       hi_start, &
       yld_form &
       )    
    real(real64), intent(inout) :: hi_end
    real(real64), intent(in) :: hi_start
    real(real64), intent(in) :: yld_form    
    hi_end = hi_start + yld_form    
  end subroutine compute_hi_end

  subroutine compute_flowering_end_cd( &
       flowering_end, &
       flowering_cd, &
       flowering, &
       hi_start, &
       crop_type &
       )
    real(real64), intent(inout) :: flowering_end
    integer(int32), intent(inout) :: flowering_cd
    integer(int32), intent(in) :: flowering
    real(real64), intent(in) :: hi_start
    integer(int32), intent(in) :: crop_type
    flowering_end = 0.
    flowering_cd = 0.
    if ( crop_type .eq. 3 ) then
       flowering_end = hi_start + flowering
       flowering_cd = flowering
    end if    
  end subroutine compute_flowering_end_cd  
       
  subroutine compute_canopy_10pct( &
       canopy_10pct, &
       emergence, &
       cc0, &
       cgc &
       )

    real(real64), intent(inout) :: canopy_10pct
    real(real64), intent(in) :: emergence
    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: cgc

    if ( cc0 /= 0. .and. cgc /= 0. ) then
       canopy_10pct = nint(emergence + log(0.1 / cc0) / cgc)
    else
       canopy_10pct = nint(emergence)
    end if
    
  end subroutine compute_canopy_10pct
  
  subroutine compute_canopy_dev_end( &
       canopy_dev_end, &
       senescence, &
       hi_start, &
       flowering, &
       determinant &
       )

    real(real64), intent(inout) :: canopy_dev_end
    real(real64), intent(in) :: senescence
    real(real64), intent(in) :: hi_start
    real(real64), intent(in) :: flowering
    integer(int32), intent(in) :: determinant
    if ( determinant == 1 ) then
       canopy_dev_end = nint(hi_start + (flowering / 2.))
    else
       canopy_dev_end = senescence
    end if
  end subroutine compute_canopy_dev_end  
    
  subroutine compute_init_cc( &
       cc0, &
       plant_pop, &
       seed_size &
       )

    real(real64), intent(inout) :: cc0
    real(real64), intent(in) :: plant_pop
    real(real64), intent(in) :: seed_size
    cc0 = nint(10000. * plant_pop * seed_size * 1E-08) / 10000.
    
  end subroutine compute_init_cc
  
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
  
  subroutine compute_root_extraction_terms( &
       sx_top, &
       sx_bot, &
       sx_top_q, &
       sx_bot_q &
       )

    real(real64), intent(inout) :: sx_top
    real(real64), intent(inout) :: sx_bot
    real(real64), intent(in) :: sx_top_q
    real(real64), intent(in) :: sx_bot_q
    real(real64) :: s1, s2, ss1, ss2, xx
    
    s1 = sx_top_q
    s2 = sx_bot_q
    if ( s1 == s2 ) then
       sx_top = s1
       sx_bot = s2
    else
       if ( sx_top_q < sx_bot_q ) then
          s1 = sx_bot_q
          s2 = sx_top_q
       end if
       xx = 3. * (s2 / (s1 - s2))
       if ( xx < 0.5 ) then
          ss1 = (4. / 3.5) * s1
          ss2 = 0.
       else
          ss1 = (xx + 3.5) * (s1 / (xx + 3.))
          ss2 = (xx - 0.5) * (s2 / xx)
       end if

       if ( sx_top_q > sx_bot_q ) then
          sx_top = ss1
          sx_bot = ss2
       else
          sx_top = ss2
          sx_bot = ss1
       end if
    end if
    
  end subroutine compute_root_extraction_terms

  subroutine compute_higc( &
       higc, &
       yld_form_cd, &
       hi0, &
       hi_ini &
       ) 

    real(real64), intent(inout) :: higc
    integer(int32), intent(in) :: yld_form_cd
    real(real64), intent(in) :: hi0
    real(real64), intent(in) :: hi_ini
    integer(int32) :: thi
    real(real64) :: hi_est

    higc = 0.001
    hi_est = 0.
    thi = yld_form_cd

    if ( thi > 0 ) then
       do while ( hi_est <= (0.98 * hi0) )
          higc = higc + 0.001
          hi_est = (hi_ini * hi0) / (hi_ini + (hi0 - hi_ini) * exp(-higc * thi))
       end do
    end if
    
    if ( hi_est >= hi0 ) then
       higc = higc - 0.001
    end if
    
  end subroutine compute_higc
  
  subroutine compute_hi_linear( &
       t_lin_switch, &
       dhi_linear, &
       hi_ini, &
       hi0, &
       higc, &
       yld_form_cd &
       )

    ! real(real64), intent(inout) :: t_lin_switch
    integer(int32), intent(inout) :: t_lin_switch
    real(real64), intent(inout) :: dhi_linear
    real(real64), intent(in) :: hi_ini
    real(real64), intent(in) :: hi0
    real(real64), intent(in) :: higc
    integer(int32), intent(in) :: yld_form_cd
    real(real64) :: ti, hi_est, hi_prev, hi_new
    ti = 0.
    hi_est = 0.
    hi_prev = hi_ini
    do while ( hi_est <= hi0 .and. ti < yld_form_cd )
       ti = ti + 1
       hi_new = (hi_ini * hi0) / (hi_ini + (hi0 - hi_ini) * exp(-higc * ti))
       hi_est = hi_new + (yld_form_cd - ti) * (hi_new - hi_prev)
       hi_prev = hi_new
    end do
    t_lin_switch = ti - 1

    if ( t_lin_switch > 0. ) then
       hi_est = (hi_ini * hi0) / (hi_ini + (hi0 - hi_ini) * exp(-higc * t_lin_switch))
    else
       hi_est = 0.
    end if
    dhi_linear = (hi0 - hi_est) / (yld_form_cd - t_lin_switch)
    
  end subroutine compute_hi_linear

end module crop_parameters

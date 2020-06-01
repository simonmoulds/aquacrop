module crop_parameters
  use types
  implicit none

  real(real64), parameter :: zero = 0.0d0
  
contains

  subroutine compute_pd_hd( &
       pd, &
       hd, &    
       planting_day, &
       harvest_day, &
       start_day &
       )

    integer(int32), intent(inout) :: pd
    integer(int32), intent(inout) :: hd
    integer(int32), intent(in) :: planting_day
    integer(int32), intent(in) :: harvest_day
    integer(int32), intent(in) :: start_day    
    pd = planting_day
    hd = harvest_day
    if ( hd < pd ) then
       hd = hd + 365
    end if
    if ( start_day > pd ) then
       pd = 0
       hd = 0
    end if
    
  end subroutine compute_pd_hd         
  
  subroutine compute_wp_adj_factor( &
       f_co2, &
       co2_current_conc, &
       co2_conc, &
       co2_refconc, &
       bsted, &
       bface, &
       fsink, &
       wp, &
       growing_season_day1 &
       )

    real(real64), intent(inout) :: f_co2
    real(real64), intent(inout) :: co2_current_conc
    real(real64), intent(in) :: co2_conc
    real(real64), intent(in) :: co2_refconc
    real(real64), intent(in) :: bsted
    real(real64), intent(in) :: bface
    real(real64), intent(in) :: fsink
    real(real64), intent(in) :: wp
    integer(int32), intent(in) :: growing_season_day1

    real(real64) :: fw, f_co2_init, ftype
    
    ! co2 weighting factor
    fw = 0.
    if ( co2_conc > co2_refconc ) then
       if (co2_conc >= 550 ) then
          fw = 1.
       else
          fw = 1. - ((550. - co2_conc) / (550. - co2_refconc))
       end if
    end if
    
    ! determine adjustment for each crop in first year of simulation
    f_co2_init = ((co2_conc / co2_refconc) / (1. + (co2_conc - co2_refconc) &
         * ((1 - fw) * bsted + fw * ((bsted * fsink) + (bface * (1 - fsink))))))

    ftype  = (40. - wp) / (40. - 20)
    if ( ftype > 1. ) then
       ftype = 1.
    else if (ftype < 0.) then
       ftype = 0.
    end if
    f_co2_init = 1 + ftype * (f_co2_init - 1)

    if ( growing_season_day1 == 1 ) then
       f_co2 = f_co2_init
       co2_current_conc = co2_conc
    end if
    
  end subroutine compute_wp_adj_factor
  
    
  subroutine switch_gdd( &
       tmin, &
       tmax, &
       planting_day, &
       harvest_day, &
       start_day, &
       emergence, &
       canopy_10pct, &
       max_rooting, &
       senescence, &
       maturity, &       
       max_canopy, &
       canopy_dev_end, &
       hi_start, &
       hi_end, &
       yld_form, &       
       max_canopy_cd, &
       canopy_dev_end_cd, &
       hi_start_cd, &
       hi_end_cd, &
       yld_form_cd, &       
       flowering_end, &
       flowering, &
       cc0, &
       ccx, &
       cgc, &
       cdc, &
       t_upp, &
       t_base, &
       crop_type, &
       gdd_method, &
       calendar_type &
       )

    real(real64), dimension(:), intent(in) :: tmin
    real(real64), dimension(:), intent(in) :: tmax
    integer(int32), intent(in) :: planting_day
    integer(int32), intent(in) :: harvest_day
    integer(int32), intent(in) :: start_day
    real(real64), intent(inout) :: emergence
    real(real64), intent(inout) :: canopy_10pct
    real(real64), intent(inout) :: max_rooting
    real(real64), intent(inout) :: senescence
    real(real64), intent(inout) :: maturity
    real(real64), intent(inout) :: max_canopy
    real(real64), intent(inout) :: canopy_dev_end
    real(real64), intent(inout) :: hi_start
    real(real64), intent(inout) :: hi_end
    real(real64), intent(inout) :: yld_form
    integer(int32), intent(in) :: max_canopy_cd
    integer(int32), intent(in) :: canopy_dev_end_cd
    integer(int32), intent(in) :: hi_start_cd
    integer(int32), intent(in) :: hi_end_cd
    integer(int32), intent(in) :: yld_form_cd
    real(real64), intent(inout) :: flowering_end
    real(real64), intent(inout) :: flowering
    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: ccx
    real(real64), intent(inout) :: cgc
    real(real64), intent(inout) :: cdc
    real(real64), intent(in) :: t_upp
    real(real64), intent(in) :: t_base
    integer(int32), intent(in) :: crop_type
    integer(int32), intent(in) :: gdd_method
    integer(int32), intent(inout) :: calendar_type
    
    integer(int32) :: pd, hd, n, i
    integer(int32) :: t_cd
    real(real64) :: t_gdd
    real(real64) :: cci
    integer(int32), allocatable, dimension(:) :: dindex
    integer(int32), allocatable, dimension(:) :: gsindex
    real(real64), allocatable, dimension(:) :: gddcum
    real(real64) :: emergence_cd
    real(real64) :: canopy_10pct_cd
    real(real64) :: max_rooting_cd
    real(real64) :: senescence_cd
    real(real64) :: maturity_cd
    real(real64) :: flowering_end_cd

    emergence_cd = emergence
    canopy_10pct_cd = canopy_10pct
    max_rooting_cd = max_rooting
    senescence_cd = senescence
    maturity_cd = maturity
    flowering_end_cd = flowering_end
    
    pd = planting_day
    hd = harvest_day
    if ( hd < pd ) then
       hd = hd + 365
    end if
    if ( start_day > pd ) then
       pd = 0
       hd = 0
    end if
    
    n = hd - start_day + 1
    allocate(dindex(n))    
    allocate(gsindex(n))
    do i = 1, n
       dindex(i) = start_day + (i - 1)
       if ( (dindex(i) .ge. pd) .and. (dindex(i) .le. hd) ) then
          gsindex(i) = 1
       else
          gsindex(i) = 0
       end if       
    end do

    allocate(gddcum(n))    
    call compute_cumulative_gdd( &
         gddcum, &
         tmin, &
         tmax, &
         t_upp, &
         t_base, &
         gdd_method &
         )

    if ( calendar_type == 1 ) then
       
       i = pd + emergence_cd
       emergence = gddcum(i)
       i = pd + canopy_10pct_cd
       canopy_10pct = gddcum(i)
       i = pd + max_rooting_cd
       max_rooting = gddcum(i)
       i = pd + senescence_cd
       senescence = gddcum(i)
       i = pd + maturity_cd
       maturity = gddcum(i)       
       i = pd + max_canopy_cd
       max_canopy = gddcum(i)
       i = pd + canopy_dev_end_cd
       canopy_dev_end = gddcum(i)
       i = pd + hi_start_cd
       hi_start = gddcum(i)
       i = pd + hi_end_cd
       hi_end = gddcum(i)
       i = pd + yld_form_cd
       yld_form = gddcum(i)       

       if ( crop_type == 3 ) then
          i = pd + flowering_end
          flowering_end = gddcum(i)
          flowering = flowering_end - hi_start
       end if

       ! convert cgc to gdd mode
       cgc = log((((0.98 * ccx) - ccx) * cc0) / (-0.25 * (ccx ** 2))) / (-(max_canopy - emergence))
       ! convert cdc to gdd mode
       t_cd = maturity_cd - senescence_cd
       if ( t_cd <= 0 ) then
          t_cd = 1
       end if
       cci = ccx * (1 - 0.05 * exp((cdc / ccx) * t_cd) - 1)
       if ( cci < 0 ) then
          cci = 0
       end if
       t_gdd = maturity - senescence
       if ( t_gdd <= 0 ) then
          t_gdd = 5
       end if
       
       cdc = (ccx / t_gdd) * log(1 + ((1 - cci / ccx) / 0.05))
       calendar_type = 2
       
    end if
    
  end subroutine switch_gdd
         
  subroutine compute_cumulative_gdd( &
       gdd_cum, &
       tmin, &
       tmax, &
       t_upp, &
       t_base, &
       gdd_method &
       )

    real(real64), dimension(:), intent(inout) :: gdd_cum
    real(real64), dimension(:), intent(in) :: tmin
    real(real64), dimension(:), intent(in) :: tmax
    real(real64), intent(in) :: t_upp
    real(real64), intent(in) :: t_base
    integer(int32), intent(in) :: gdd_method    
    real(real64), allocatable, dimension(:) :: tmean, gdd
    real(real64) :: tmn, tmx
    integer(int32) :: i, n

    n = size(gdd_cum)
    allocate(tmean(n))
    allocate(gdd(n))

    do i = 1, n
       tmn = tmin(i)
       tmx = tmax(i)
       if ( gdd_method == 1 ) then
          tmean(i) = (tmx + tmn) / 2
          if ( tmean(i) > t_upp ) then
             tmean(i) = t_upp
          else if ( tmean(i) < t_base ) then
             tmean(i) = t_base
          end if

       else if ( gdd_method == 2 .or. gdd_method == 3 ) then
          if ( tmx < t_base ) then
             tmx = t_base
          else if ( tmx > t_upp ) then
             tmx = t_upp
          end if

          if ( gdd_method == 2 .and. tmn < t_base ) then
             tmn = t_base
          else if ( tmn > t_upp ) then
             tmn = t_upp
          end if          
          tmean(i) = (tmx + tmn) / 2
       end if
       gdd(i) = tmean(i) - t_base

       if ( i > 1 ) then
          gdd_cum(i) = gdd_cum(i-1) + gdd(i)
       else
          gdd_cum(i) = gdd(i)
       end if
       
    end do
    
  end subroutine compute_cumulative_gdd
  
  ! subroutine compute_crop_calendar_type1( &
  !      max_canopy_cd, &
  !      canopy_dev_end_cd, &
  !      hi_start_cd, &
  !      hi_end_cd, &
  !      yld_form_cd, &
  !      flowering_cd, &
  !      max_canopy, &
  !      canopy_dev_end, &
  !      hi_start, &
  !      hi_end, &
  !      yld_form, &
  !      flowering &
  !      )

  !   integer(int32), intent(inout) :: max_canopy_cd
  !   integer(int32), intent(inout) :: canopy_dev_end_cd
  !   integer(int32), intent(inout) :: hi_start_cd
  !   integer(int32), intent(inout) :: hi_end_cd
  !   integer(int32), intent(inout) :: yld_form_cd
  !   integer(int32), intent(inout) :: flowering_cd
  !   real(real64), intent(in) :: max_canopy
  !   real(real64), intent(in) :: canopy_dev_end
  !   real(real64), intent(in) :: hi_start
  !   real(real64), intent(in) :: hi_end
  !   real(real64), intent(in) :: yld_form
  !   real(real64), intent(in) :: flowering    
  !   max_canopy_cd = max_canopy
  !   canopy_dev_end_cd = canopy_dev_end
  !   hi_start_cd = hi_start
  !   hi_end_cd = hi_end
  !   yld_form_cd = yld_form
  !   flowering_cd = flowering
    
  ! end subroutine compute_crop_calendar_type1


  subroutine compute_crop_calendar_type2( &
       tmin, &
       tmax, &
       planting_day, &
       harvest_day, &
       start_day, &
       max_canopy, &
       canopy_dev_end, &
       hi_start, &
       hi_end, &
       max_canopy_cd, &
       canopy_dev_end_cd, &
       hi_start_cd, &
       hi_end_cd, &
       yld_form_cd, &
       flowering_cd, &
       flowering_end, &
       t_upp, &
       t_base, &
       crop_type, &
       gdd_method, &
       calendar_type, &
       growing_season_day1, &
       simulation_day1 &
       )

    real(real64), dimension(:), intent(in) :: tmin
    real(real64), dimension(:), intent(in) :: tmax
    integer(int32), intent(in) :: planting_day
    integer(int32), intent(in) :: harvest_day
    integer(int32), intent(in) :: start_day
    real(real64), intent(in) :: max_canopy
    real(real64), intent(in) :: canopy_dev_end
    real(real64), intent(in) :: hi_start
    real(real64), intent(in) :: hi_end
    integer(int32), intent(inout) :: max_canopy_cd
    integer(int32), intent(inout) :: canopy_dev_end_cd
    integer(int32), intent(inout) :: hi_start_cd
    integer(int32), intent(inout) :: hi_end_cd
    integer(int32), intent(inout) :: yld_form_cd
    integer(int32), intent(inout) :: flowering_cd
    real(real64), intent(in) :: flowering_end
    real(real64), intent(in) :: t_upp
    real(real64), intent(in) :: t_base
    integer(int32), intent(in) :: crop_type
    integer(int32), intent(in) :: gdd_method
    integer(int32), intent(in) :: calendar_type
    integer(int32), intent(in) :: growing_season_day1
    integer(int32), intent(in) :: simulation_day1
    
    integer(int32) :: pd, hd, n, i
    real(real64) :: cci
    
    integer(int32) :: max_canopy_idx
    integer(int32) :: canopy_dev_end_idx
    integer(int32) :: hi_start_idx
    integer(int32) :: hi_end_idx    
    integer(int32) :: flowering_end_idx
    
    integer(int32), allocatable, dimension(:) :: dindex
    integer(int32), allocatable, dimension(:) :: gsindex
    integer(int32), allocatable, dimension(:) :: fill
    real(real64), allocatable, dimension(:) :: gddcum

    ! correct planting day/harvest day to ensure that harvest
    ! is *after* planting
    pd = planting_day
    hd = harvest_day
    if ( hd < pd ) then
       hd = hd + 365
    end if
    if ( start_day > pd ) then
       pd = 0
       hd = 0
    end if

    ! allocate array variables
    n = hd - start_day + 1
    allocate(dindex(n))    
    allocate(gsindex(n))
    allocate(gddcum(n))    
    allocate(fill(n))
    fill = 999

    ! day index, growing season index
    do i = 1, n
       dindex(i) = start_day + (i - 1)
       if ( (dindex(i) .ge. pd) .and. (dindex(i) .le. hd) ) then
          gsindex(i) = 1
       else
          gsindex(i) = 0
       end if       
    end do

    ! compute cumulative gdd
    call compute_cumulative_gdd( &
         gddcum, &
         tmin, &
         tmax, &
         t_upp, &
         t_base, &
         gdd_method &
         )

    ! get the index of the first day for which gddcum exceeds max_canopy etc.
    max_canopy_idx = minval(pack(dindex, gddcum>max_canopy, fill))
    canopy_dev_end_idx = minval(pack(dindex, gddcum>canopy_dev_end, fill))
    hi_start_idx = minval(pack(dindex, gddcum>hi_start, fill))
    hi_end_idx = minval(pack(dindex, gddcum>hi_end, fill))
    flowering_end_idx = minval(pack(dindex, gddcum>flowering_end, fill))
    
    if ( growing_season_day1 == 1 .or. simulation_day1 == 1) then
       max_canopy_cd = max_canopy_idx - pd + 1
       canopy_dev_end_cd = canopy_dev_end_idx - pd + 1
       hi_start_cd = hi_start_idx - pd + 1
       hi_end_cd = hi_end_idx - pd + 1
       if ( crop_type == 3 ) then
          flowering_cd = (flowering_end_idx - pd + 1) - hi_start_cd
       end if
       yld_form_cd = hi_end_cd - hi_start_cd
    end if
    
  end subroutine compute_crop_calendar_type2    
    
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

    ! ! TEST
    ! if ( harvest_date_adj < planting_date_adj ) then
    !    harvest_date_adj = harvest_date_adj + 365
    ! end if    
    
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

module crop_parameters_w
  use types
  use crop_parameters
  implicit none

contains

  subroutine update_crop_parameters_w( &
       higc, &
       t_lin_switch, &
       dhi_linear, &
       yld_form_cd, &
       hi0, &
       hi_ini, &
       n_farm, n_crop, n_cell &
       )

    integer(int32) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: higc
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: t_lin_switch
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: dhi_linear
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: yld_form_cd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi0
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_ini
    integer(int32) :: i, j, k

    do i = 1, n_cell
       do j = 1, n_crop
          do k = 1, n_farm
             call update_crop_parameters( &
                  higc(k,j,i), &
                  t_lin_switch(k,j,i), &
                  dhi_linear(k,j,i), &
                  yld_form_cd(k,j,i), &
                  hi0(k,j,i), &
                  hi_ini(k,j,i) &
                  )
          end do
       end do
    end do
    
  end subroutine update_crop_parameters_w
  
               
    
  subroutine compute_crop_parameters_w( &
       cc0, &
       sx_top, &
       sx_bot, &
       planting_date_adj, &
       harvest_date_adj, &
       canopy_dev_end, &
       canopy_10pct, &
       max_canopy, &
       hi_end, &
       flowering_end, &
       flowering_cd, &
       higc, &
       t_lin_switch, &
       dhi_linear, &
       plant_pop, &
       seed_size, &
       sx_top_q, &
       sx_bot_q, &
       planting_date, &
       harvest_date, &
       day_of_year, &
       time_step, &
       leap_year, &
       senescence, &
       hi_start, &
       flowering, &
       determinant, &
       emergence, &
       cgc, &
       ccx, &
       yld_form, &
       crop_type, &
       yld_form_cd, &
       hi0, &
       hi_ini, &
       n_farm, n_crop, n_cell &       
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc0
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: sx_top
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: sx_bot
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: planting_date_adj
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: harvest_date_adj
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: canopy_dev_end
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: canopy_10pct
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: max_canopy
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: hi_end
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: flowering_end
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: flowering_cd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: higc
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: t_lin_switch
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: dhi_linear
    
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: plant_pop
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: seed_size
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: sx_top_q
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: sx_bot_q
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: planting_date
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: harvest_date
    integer(int32), intent(in) :: day_of_year
    integer(int32), intent(in) :: time_step
    integer(int32), intent(in) :: leap_year
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: senescence
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: flowering
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: determinant
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: emergence
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cgc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ccx
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: yld_form    
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: crop_type
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: yld_form_cd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi0
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_ini
    
    integer(int32) :: i, j, k
    
    do i = 1, n_cell
       do j = 1, n_crop
          do k = 1, n_farm
             call compute_crop_parameters( &
                  cc0(k,j,i), &
                  sx_top(k,j,i), &
                  sx_bot(k,j,i), &
                  planting_date_adj(k,j,i), &
                  harvest_date_adj(k,j,i), &
                  canopy_dev_end(k,j,i), &
                  canopy_10pct(k,j,i), &
                  max_canopy(k,j,i), &
                  hi_end(k,j,i), &
                  flowering_end(k,j,i), &
                  flowering_cd(k,j,i), &
                  higc(k,j,i), &
                  t_lin_switch(k,j,i), &
                  dhi_linear(k,j,i), &
                  plant_pop(k,j,i), &
                  seed_size(k,j,i), &
                  sx_top_q(k,j,i), &
                  sx_bot_q(k,j,i), &
                  planting_date(k,j,i), &
                  harvest_date(k,j,i), &
                  day_of_year, &
                  time_step, &
                  leap_year, &
                  senescence(k,j,i), &
                  hi_start(k,j,i), &
                  flowering(k,j,i), &
                  determinant(k,j,i), &
                  emergence(k,j,i), &
                  cgc(k,j,i), &
                  ccx(k,j,i), &
                  yld_form(k,j,i), &
                  crop_type(k,j,i), &
                  yld_form_cd(k,j,i), &
                  hi0(k,j,i), &
                  hi_ini(k,j,i) &
                  )
          end do          
       end do
    end do
  end subroutine compute_crop_parameters_w
  
  subroutine switch_gdd_w( &
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
       calendar_type, &
       n_day, &
       n_farm, n_crop, n_cell &       
       )

    integer(int32), intent(in) :: n_day
    integer(int32), intent(in) :: n_farm, n_crop, n_cell    
    real(real64), dimension(n_day, n_cell), intent(in) :: tmin
    real(real64), dimension(n_day, n_cell), intent(in) :: tmax
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: planting_day
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: harvest_day    
    integer(int32), intent(in) :: start_day
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: emergence
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: canopy_10pct
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: max_rooting
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: senescence
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: maturity
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: max_canopy
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: canopy_dev_end
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: hi_start
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: hi_end
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: yld_form
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: max_canopy_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: canopy_dev_end_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_end_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: yld_form_cd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: flowering_end
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: flowering
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc0
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ccx
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cgc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cdc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_upp
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_base
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: crop_type
    integer(int32), intent(in) :: gdd_method
    integer(int32), intent(inout) :: calendar_type
    integer(int32) :: i, j, k
    
    do i = 1, n_cell
       do j = 1, n_crop
          do k = 1, n_farm
             call switch_gdd( &
                  tmin(:,i), &
                  tmax(:,i), &
                  planting_day(k,j,i), &
                  harvest_day(k,j,i), &
                  start_day, &
                  emergence(k,j,i), &
                  canopy_10pct(k,j,i), &
                  max_rooting(k,j,i), &
                  senescence(k,j,i), &
                  maturity(k,j,i), &       
                  max_canopy(k,j,i), &
                  canopy_dev_end(k,j,i), &
                  hi_start(k,j,i), &
                  hi_end(k,j,i), &
                  yld_form(k,j,i), &       
                  max_canopy_cd(k,j,i), &
                  canopy_dev_end_cd(k,j,i), &
                  hi_start_cd(k,j,i), &
                  hi_end_cd(k,j,i), &
                  yld_form_cd(k,j,i), &       
                  flowering_end(k,j,i), &
                  flowering(k,j,i), &
                  cc0(k,j,i), &
                  ccx(k,j,i), &
                  cgc(k,j,i), &
                  cdc(k,j,i), &
                  t_upp(k,j,i), &
                  t_base(k,j,i), &
                  crop_type(k,j,i), &
                  gdd_method, &
                  calendar_type &
                  )
          end do          
       end do
    end do
    
  end subroutine switch_gdd_w  
  
  subroutine compute_crop_calendar_type2_w( &
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
       simulation_day1, &
       n_day, &
       n_farm, n_crop, n_cell &       
       )

    integer(int32), intent(in) :: n_day, n_farm, n_crop, n_cell
    real(real64), dimension(n_day, n_cell), intent(in) :: tmin
    real(real64), dimension(n_day, n_cell), intent(in) :: tmax
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: planting_day
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: harvest_day    
    integer(int32), intent(in) :: start_day    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: max_canopy
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: canopy_dev_end
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_end
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: max_canopy_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: canopy_dev_end_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: hi_start_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: hi_end_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: yld_form_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: flowering_cd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: flowering_end
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_upp
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_base
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: crop_type    
    integer(int32), intent(in) :: gdd_method
    integer(int32), intent(in) :: calendar_type    
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season_day1
    
    integer(int32), intent(in) :: simulation_day1

    integer(int32) :: i, j, k

    do i = 1, n_cell
       do j = 1, n_crop
          do k = 1, n_farm
             call compute_crop_calendar_type2( &
                  tmin(:,i), &
                  tmax(:,i), &
                  planting_day(k,j,i), &
                  harvest_day(k,j,i), &
                  start_day, &
                  max_canopy(k,j,i), &
                  canopy_dev_end(k,j,i), &
                  hi_start(k,j,i), &
                  hi_end(k,j,i), &
                  max_canopy_cd(k,j,i), &
                  canopy_dev_end_cd(k,j,i), &
                  hi_start_cd(k,j,i), &
                  hi_end_cd(k,j,i), &
                  yld_form_cd(k,j,i), &
                  flowering_cd(k,j,i), &
                  flowering_end(k,j,i), &
                  t_upp(k,j,i), &
                  t_base(k,j,i), &
                  crop_type(k,j,i), &
                  gdd_method, &
                  calendar_type, &
                  growing_season_day1(k,j,i), &
                  simulation_day1 &
                  )
          end do          
       end do
    end do
    
  end subroutine compute_crop_calendar_type2_w
  
  
  subroutine compute_wp_adj_factor_w( &
       f_co2, &
       co2_current_conc, &
       co2_conc, &
       co2_refconc, &
       bsted, &
       bface, &
       fsink, &
       wp, &
       growing_season_day1, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: f_co2
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: co2_current_conc
    real(real64), dimension(n_cell), intent(in) :: co2_conc
    real(real64), intent(in) :: co2_refconc
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: bsted
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: bface
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: fsink
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: wp
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season_day1
    integer(int32) :: i, j, k

    do i = 1, n_cell
       do j = 1, n_crop
          do k = 1, n_farm
             call compute_wp_adj_factor( &
                  f_co2(k,j,i), &
                  co2_current_conc(k,j,i), &
                  co2_conc(i), &
                  co2_refconc, &
                  bsted(k,j,i), &
                  bface(k,j,i), &
                  fsink(k,j,i), &
                  wp(k,j,i), &
                  growing_season_day1(k,j,i) &
                  )
          end do          
       end do
    end do
    
  end subroutine compute_wp_adj_factor_w                   
  
  ! subroutine compute_flowering_end_cd_w( &
  !      flowering_end, &
  !      flowering_cd, &
  !      flowering, &
  !      hi_start, &
  !      crop_type, &
  !      n_farm, n_crop, n_cell &
  !      )

  !   integer(int32), intent(in) :: n_farm, n_crop, n_cell
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: flowering_end
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: flowering_cd
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: flowering
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: crop_type
  !   integer(int32) :: i, j, k

  !   do i = 1, n_cell
  !      do j = 1, n_crop
  !         do k = 1, n_farm
  !            call compute_flowering_end_cd( &
  !                 flowering_end(k,j,i), &
  !                 flowering_cd(k,j,i), &
  !                 flowering(k,j,i), &
  !                 hi_start(k,j,i), &
  !                 crop_type(k,j,i) &                  
  !                 )
  !         end do          
  !      end do
  !   end do
    
  ! end subroutine compute_flowering_end_cd_w  
    
  ! subroutine compute_hi_end_w( &
  !      hi_end, &
  !      hi_start, &
  !      yld_form, &
  !      n_farm, n_crop, n_cell &
  !      )

  !   integer(int32), intent(in) :: n_farm, n_crop, n_cell
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: hi_end
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: yld_form
  !   integer(int32) :: i, j, k
    
  !   do i = 1, n_cell
  !      do j = 1, n_crop
  !         do k = 1, n_farm
  !            call compute_hi_end( &
  !                 hi_end(k,j,i), &
  !                 hi_start(k,j,i), &
  !                 yld_form(k,j,i) &
  !                 )
  !         end do          
  !      end do
  !   end do
    
  ! end subroutine compute_hi_end_w
  
  ! subroutine compute_max_canopy_w( &
  !      max_canopy, &
  !      emergence, &
  !      ccx, &
  !      cc0, &
  !      cgc, &
  !      n_farm, n_crop, n_cell &
  !      )

  !   integer(int32), intent(in) :: n_farm, n_crop, n_cell
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: max_canopy
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: emergence
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ccx
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc0
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cgc
  !   integer(int32) :: i, j, k

  !   do i = 1, n_cell
  !      do j = 1, n_crop
  !         do k = 1, n_farm
  !            call compute_max_canopy( &
  !                 max_canopy(k,j,i), &
  !                 emergence(k,j,i), &
  !                 ccx(k,j,i), &
  !                 cc0(k,j,i), &
  !                 cgc(k,j,i) &
  !                 )
  !         end do          
  !      end do
  !   end do
  ! end subroutine compute_max_canopy_w
  
             
  ! subroutine compute_canopy_10pct_w( &
  !      canopy_10pct, &
  !      emergence, &
  !      cc0, &
  !      cgc, &
  !      n_farm, n_crop, n_cell &
  !      )
  !   integer(int32), intent(in) :: n_farm, n_crop, n_cell
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: canopy_10pct
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: emergence
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc0
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cgc
  !   integer(int32) :: i, j, k
    
  !   do i = 1, n_cell
  !      do j = 1, n_crop
  !         do k = 1, n_farm
  !            call compute_canopy_10pct( &
  !                 canopy_10pct(k,j,i), &
  !                 emergence(k,j,i), &
  !                 cc0(k,j,i), &
  !                 cgc(k,j,i) &
  !                 )
  !         end do          
  !      end do
  !   end do
    
  ! end subroutine compute_canopy_10pct_w
  
       
  ! subroutine compute_canopy_dev_end_w( &
  !      canopy_dev_end, &
  !      senescence, &
  !      hi_start, &
  !      flowering, &
  !      determinant, &
  !      n_farm, n_crop, n_cell &
  !      )
    
  !   integer(int32), intent(in) :: n_farm, n_crop, n_cell
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: canopy_dev_end
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: senescence
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: flowering
  !   integer(int32), intent(in) :: determinant
  !   integer(int32) :: i, j, k
    
  !   do i = 1, n_cell
  !      do j = 1, n_crop
  !         do k = 1, n_farm
  !            call compute_canopy_dev_end( &
  !                 canopy_dev_end(k,j,i), &
  !                 senescence(k,j,i), &
  !                 hi_start(k,j,i), &
  !                 flowering(k,j,i), &
  !                 determinant &
  !                 )
  !         end do
  !      end do
  !   end do    
    
  ! end subroutine compute_canopy_dev_end_w             
    
  ! subroutine compute_init_cc_w( &
  !      cc0, &
  !      plant_pop, &
  !      seed_size, &
  !      n_farm, n_crop, n_cell &
  !      )

  !   integer(int32), intent(in) :: n_farm, n_crop, n_cell
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc0
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: plant_pop
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: seed_size
  !   integer(int32) :: i, j, k

  !   do i = 1, n_cell
  !      do j = 1, n_crop
  !         do k = 1, n_farm
  !            call compute_init_cc( &
  !                 cc0(k,j,i), &
  !                 plant_pop(k,j,i), &
  !                 seed_size(k,j,i) &
  !                 )
  !         end do          
  !      end do
  !   end do
  ! end subroutine compute_init_cc_w
  
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
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: planting_date_adj
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: harvest_date_adj
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: planting_date
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: harvest_date
    integer(int32), intent(in) :: day_of_year
    integer(int32), intent(in) :: time_step
    integer(int32), intent(in) :: leap_year
    integer(int32) :: i, j, k
    
    do i = 1, n_cell
       do j = 1, n_crop
          do k = 1, n_farm
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
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: growing_season_index
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: growing_season_day_one
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: dap    
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: pd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hd    
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: crop_dead
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: crop_mature
    integer(int32), intent(in) :: doy
    integer(int32), intent(in) :: time_step
    integer(int32), intent(in) :: year_start_num
    integer(int32), intent(in) :: end_time_num
    integer(int32) :: i, j, k
    
    do i = 1, n_cell
       do j = 1, n_crop
          do k = 1, n_farm
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

!   subroutine compute_root_extraction_terms_w( &
!        sx_top, &
!        sx_bot, &
!        sx_top_q, &
!        sx_bot_q, &
!        n_farm, n_crop, n_cell &
!        )

!     integer(int32) :: n_farm, n_crop, n_cell
!     real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: sx_top
!     real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: sx_bot
!     real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: sx_top_q
!     real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: sx_bot_q
!     integer(int32) :: i, j, k
    
!     do i = 1, n_cell
!        do j = 1, n_crop
!           do k = 1, n_farm
!              call compute_root_extraction_terms( &
!                   sx_top(k,j,i), &
!                   sx_bot(k,j,i), &
!                   sx_top_q(k,j,i), &
!                   sx_bot_q(k,j,i) &
!                   )
!           end do
!        end do
!     end do    
    
!   end subroutine compute_root_extraction_terms_w

!   subroutine compute_hi_linear_w( &
!        t_lin_switch, &
!        dhi_linear, &
!        hi_ini, &
!        hi0, &
!        higc, &
!        yld_form_cd, &
!        n_farm, n_crop, n_cell &
!        )

!     integer(int32) :: n_farm, n_crop, n_cell
!     integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: t_lin_switch
!     real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: dhi_linear
!     real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_ini
!     real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi0
!     real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: higc
!     integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: yld_form_cd
!     integer(int32) :: i, j, k

!     do i = 1, n_cell
!        do j = 1, n_crop
!           do k = 1, n_farm
!              call compute_hi_linear( &
!                   t_lin_switch(k,j,i), &
!                   dhi_linear(k,j,i), &
!                   hi_ini(k,j,i), &
!                   hi0(k,j,i), &
!                   higc(k,j,i), &
!                   yld_form_cd(k,j,i) &
!                   )
!           end do          
!        end do
!     end do

!   end subroutine compute_hi_linear_w

!   subroutine compute_higc_w( &
!        higc, &
!        yld_form_cd, &
!        hi0, &
!        hi_ini, &
!        n_farm, n_crop, n_cell &
!        )

!     integer(int32) :: n_farm, n_crop, n_cell
!     real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: higc
!     integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: yld_form_cd
!     real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi0
!     real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_ini
!     integer(int32) :: i, j, k
    
!     do i = 1, n_cell
!        do j = 1, n_crop
!           do k = 1, n_farm
!              call compute_higc( &
!                   higc(k,j,i), &
!                   yld_form_cd(k,j,i), &
!                   hi0(k,j,i), &
!                   hi_ini(k,j,i) &
!                   )
!           end do          
!        end do
!     end do
    
!   end subroutine compute_higc_w
                  
end module crop_parameters_w

  
    
  

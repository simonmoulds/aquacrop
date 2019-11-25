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

  subroutine compute_root_extraction_terms_w( &
       sx_top, &
       sx_bot, &
       sx_top_q, &
       sx_bot_q, &
       n_farm, n_crop, n_cell &
       )

    integer(int32) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: sx_top
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: sx_bot
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: sx_top_q
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: sx_bot_q
    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call compute_root_extraction_terms( &
                  sx_top(k,j,i), &
                  sx_bot(k,j,i), &
                  sx_top_q(k,j,i), &
                  sx_bot_q(k,j,i) &
                  )
          end do
       end do
    end do
  end subroutine compute_root_extraction_terms_w

  subroutine compute_hi_linear_w( &
       t_lin_switch, &
       dhi_linear, &
       hi_ini, &
       hi0, &
       higc, &
       yld_form_cd, &
       n_farm, n_crop, n_cell &
       )

    integer(int32) :: n_farm, n_crop, n_cell
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: t_lin_switch
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: dhi_linear
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_ini
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi0
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: higc
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: yld_form_cd
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call compute_hi_linear( &
                  t_lin_switch(k,j,i), &
                  dhi_linear(k,j,i), &
                  hi_ini(k,j,i), &
                  hi0(k,j,i), &
                  higc(k,j,i), &
                  yld_form_cd(k,j,i) &
                  )
          end do
       end do
    end do
  end subroutine compute_hi_linear_w

  subroutine compute_higc_w( &
       higc, &
       yld_form_cd, &
       hi0, &
       hi_ini, &
       n_farm, n_crop, n_cell &
       )

    integer(int32) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: higc
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: yld_form_cd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi0
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_ini
    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call compute_higc( &
                  higc(k,j,i), &
                  yld_form_cd(k,j,i), &
                  hi0(k,j,i), &
                  hi_ini(k,j,i) &
                  )
          end do
       end do
    end do
  end subroutine compute_higc_w
  
                  
end module crop_parameters_w

  
    
  

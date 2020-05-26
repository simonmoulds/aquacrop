module irrigation_w
  use types
  use irrigation, only: update_irrigation
  implicit none

contains

  subroutine update_irrigation_w( &
       irr_method, &
       irr, &
       irr_cum, &
       irr_net_cum, &
       smt1, &
       smt2, &
       smt3, &
       smt4, &
       irr_scheduled, &
       app_eff, &
       z_root, &
       z_min, &
       taw, &
       dr, &
       thrz_fc, &
       thrz_act, &
       prec, &
       runoff, &
       et_pot, &
       max_irr, &
       irr_interval, &
       dap, &
       growth_stage, &
       growing_season_day1, &
       growing_season, &
       n_farm, &
       n_crop, &
       n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: irr_method
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr_cum
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr_net_cum
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt1
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt2
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt3
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: smt4
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: irr_scheduled    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: app_eff
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_root 
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_min 
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: taw 
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: dr
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: thrz_fc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: thrz_act 
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: prec
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: runoff
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: et_pot
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: max_irr
    
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: irr_interval    
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: dap
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growth_stage    
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season_day1
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season

    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             
             call update_irrigation( &
                  irr_method(k,j,i), &
                  irr(k,j,i), &
                  irr_cum(k,j,i), &
                  irr_net_cum(k,j,i), &
                  smt1(k,j,i), &
                  smt2(k,j,i), &
                  smt3(k,j,i), &
                  smt4(k,j,i), &
                  irr_scheduled(k,j,i), &
                  app_eff(k,j,i), &
                  z_root(k,j,i), &
                  z_min(k,j,i), &
                  taw(k,j,i), &
                  dr(k,j,i), &
                  thrz_fc(k,j,i), &
                  thrz_act(k,j,i), &
                  prec(k,j,i), &
                  runoff(k,j,i), &
                  et_pot(k,j,i), &
                  max_irr(k,j,i), &
                  irr_interval(k,j,i), &
                  dap(k,j,i), &
                  growth_stage(k,j,i), &
                  growing_season_day1(k,j,i), &
                  growing_season(k,j,i) &
                  )
             
          end do
       end do
    end do
  end subroutine update_irrigation_w
end module irrigation_w
    
       
       

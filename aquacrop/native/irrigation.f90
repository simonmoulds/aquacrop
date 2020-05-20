module irrigation
  use types
  implicit none

contains

  subroutine update_irrigation( &
       irr_method, &            ! smt = 1; fixed interval = 2; schedule = 3; net = 4
       irr, &
       irr_cum, &
       irr_net_cum, &           ! TODO - move to pre_irr
       smt, &                   ! soil moisture threshold
       irr_scheduled, &
       app_eff, &               ! irrigation application efficiency
       z_root, &
       z_min, &
       taw, &                   ! total available water
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
       growing_season &
       )

    integer(int32), intent(in) :: irr_method
    
    real(real64), intent(inout) :: irr
    real(real64), intent(inout) :: irr_cum
    real(real64), intent(inout) :: irr_net_cum
    
    real(real64), dimension(:), intent(in) :: smt
    real(real64), intent(inout) :: irr_scheduled    
    real(real64), intent(in) :: app_eff
    
    real(real64), intent(in) :: z_root 
    real(real64), intent(in) :: z_min 
    real(real64), intent(in) :: taw 
    real(real64), intent(in) :: dr
    real(real64), intent(in) :: thrz_fc
    real(real64), intent(in) :: thrz_act 
    real(real64), intent(in) :: prec
    real(real64), intent(in) :: runoff
    real(real64), intent(in) :: et_pot
    real(real64), intent(in) :: max_irr
    
    integer(int32), intent(in) :: irr_interval    
    integer(int32), intent(in) :: dap
    integer(int32), intent(in) :: growth_stage    
    integer(int32), intent(in) :: growing_season_day1
    integer(int32), intent(in) :: growing_season

    real(real64) :: rootdepth
    real(real64) :: abv_fc
    real(real64) :: wc_adj
    real(real64) :: smt_i       ! soil moisture threshold for current growth stage
    real(real64) :: dr_adj
    real(real64) :: irr_thr     ! threshold for irrigation
    real(real64) :: irr_req     ! required irrigation
    real(real64) :: eff_adj
    integer(int32) :: remainder !
    integer(int32) :: n_days    ! number of days since planting, minus one
    integer(int32) :: irrigate  ! flag to irrigate
    
    if ( growing_season == 1) then

       if ( irr_method == 0 ) then
          ! Rainfed - no irigation
          irr = 0
          
       else if ( irr_method == 1 .or. irr_method == 2 ) then
          
          ! Determine adjustment for inflows/outflows on current day
          rootdepth = max(z_root, z_min)
          if ( thrz_act > thrz_fc ) then
             abv_fc = (thrz_act - thrz_fc) * 1000. * rootdepth
          else
             abv_fc = 0.
          end if
          wc_adj = et_pot - prec + runoff - abv_fc
          
          dr_adj = dr + wc_adj
          if (dr_adj < 0.) then
             dr_adj = 0.
          end if

          ! Set irrigate flag to zero
          irrigate = 0          
          if ( irr_method == 1 ) then
             
             ! Soil moisture threshold
             smt_i = smt(growth_stage)
             irr_thr = (1. - smt_i / 100.) * taw
             ! Check if depletion exceeds threshold
             if ( dr_adj > irr_thr ) then
                irrigate = 1
             end if
             
          else if ( irr_method == 2 ) then
             
             ! Fixed interval
             n_days = dap - 1
             ! mod(n,m) gives remainder when n is divided by m
             remainder = mod(n_days, irr_interval)
             if ( remainder == 0 ) then
                irrigate = 1
             end if

          end if

          if ( irrigate == 1 ) then
             ! Irrigation occurs
             irr_req = max(0., dr_adj)
             ! Adjust for application efficiency
             eff_adj = ((100. - app_eff) + 100.) / 100.
             irr_req = irr_req * eff_adj
             irr = min(irr_req, max_irr)
          else
             irr = 0.
          end if          

       else if ( irr_method == 3 ) then
          irr = irr_scheduled
          
       else if ( irr_method == 4 ) then
          ! Net irrigation is computed after transpiration, so
          ! here we set irrigation to zero
          irr = 0.
       end if
       irr_cum = irr_cum + irr
    else
       irr = 0.
       irr_cum = 0.
       irr_net_cum = 0.         ! TODO - remove from this module
    end if

  end subroutine update_irrigation

end module irrigation

module growth_stage
  use types
  implicit none

contains
  
  subroutine update_growth_stage( &
       growth_stage, &
       canopy_10pct, &
       max_canopy, &
       senescence, &
       gdd_cum, &
       dap, &
       delayed_cds, &
       delayed_gdds, &       
       calendar_type, &
       growing_season &
       )

    integer(int32), intent(inout) :: growth_stage
    real(real64), intent(in) :: canopy_10pct
    real(real64), intent(in) :: max_canopy
    real(real64), intent(in) :: senescence
    real(real64), intent(in) :: gdd_cum
    integer(int32), intent(in) :: dap
    real(real64), intent(in) :: delayed_cds
    real(real64), intent(in) :: delayed_gdds
    integer(int32), intent(in) :: calendar_type
    integer(int32), intent(in) :: growing_season
    real(real64) :: t_adj
    
    if ( growing_season == 1 ) then
       if ( calendar_type == 1 ) then
          t_adj = dap - delayed_cds
       else if ( calendar_type == 2 ) then
          t_adj = gdd_cum - delayed_gdds
       end if

       if ( t_adj <= canopy_10pct ) then
          growth_stage = 1
       else if ( t_adj <= max_canopy ) then
          growth_stage = 2
       else if ( t_adj <= senescence ) then
          growth_stage = 3
       else if ( t_adj >= senescence ) then
          growth_stage = 4
       end if
       
    else
       growth_stage = 0       
    end if
    
  end subroutine update_growth_stage
end module growth_stage
  

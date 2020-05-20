module crop_yield
  use types
  implicit none

contains

  subroutine update_crop_yield( &
       yield, &
       crop_mature, &
       maturity, &
       b, &
       hi_adj, &
       gdd_cum, &
       growing_season, &
       growing_season_day1, &
       calendar_type, &
       dap, &       
       delayed_cds, &
       delayed_gdds &
       )

    real(real64), intent(inout) :: yield
    integer(int32), intent(inout) :: crop_mature
    real(real64), intent(in) :: maturity
    real(real64), intent(in) :: b
    real(real64), intent(in) :: hi_adj
    real(real64), intent(in) :: gdd_cum
    integer(int32), intent(in) :: growing_season
    integer(int32), intent(in) :: growing_season_day1
    integer(int32), intent(in) :: calendar_type
    integer(int32), intent(in) :: dap
    integer(int32), intent(in) :: delayed_cds
    real(real64), intent(in) :: delayed_gdds

    if ( growing_season_day1 == 1 ) then
       crop_mature = 0       
    end if

    if ( growing_season == 1 ) then
       
       yield = (b / 100.) * hi_adj
       if ( calendar_type == 1 ) then
          if ( (dap - delayed_cds) >= maturity ) then
             crop_mature = 1
          else
             crop_mature = 0
          end if
          
       else if ( calendar_type == 2 ) then
          if ( (gdd_cum - delayed_gdds) >= maturity ) then
             crop_mature = 1
          else
             crop_mature = 0
          end if
       end if       
    else
       yield = 0
    end if
    
  end subroutine update_crop_yield
  
end module crop_yield


    

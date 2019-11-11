module gdd
  use types
  implicit none

contains

  function mean_temp( &
       method, &
       t_max, &
       t_min, &
       t_base, &
       t_upp) result(t_mean)

    integer(int32), intent(in) :: method
    real(real64), intent(in) :: t_max
    real(real64), intent(in) :: t_min
    real(real64), intent(in) :: t_base
    real(real64), intent(in) :: t_upp
    real(real64) :: t_max2
    real(real64) :: t_min2
    real(real64) :: t_mean

    if ( method == 1 ) then

       t_mean = (t_max + t_min) / 2.
       if ( t_mean > t_upp ) then
          t_mean = t_upp
       end if
       
       if ( t_mean < t_base) then
          t_mean = t_base
       end if
       
    else if ( method == 2 ) then
       t_max2 = t_max
       t_min2 = t_min
       if ( t_max2 > t_upp ) then
          t_max2 = t_upp
       end if

       if ( t_max2 < t_base ) then
          t_max2 = t_base
       end if

       if ( t_min2 > t_upp ) then
          t_min2 = t_upp
       end if
       
       if ( t_min2 < t_base ) then
          t_min2 = t_base
       end if
       t_mean = (t_max2 + t_min2) / 2.

    else if ( method == 3 ) then
       t_max2 = t_max
       t_min2 = t_min
       if ( t_max2 > t_upp ) then
          t_max2 = t_upp
       end if
       
       if ( t_max2 < t_base ) then
          t_max2 = t_base
       end if

       if ( t_min2 > t_upp ) then
          t_min2 = t_upp
       end if

       t_mean = (t_max2 + t_min2) / 2.
       if ( t_mean < t_base ) then
          t_mean = t_base
       end if
       
    end if
    
  end function mean_temp
  
  subroutine update_gdd( &
       gdd, &
       gdd_cum, &
       gdd_method, &
       t_max, &
       t_min, &
       t_base, &
       t_upp, &
       growing_season &
       )

    real(real64), intent(inout) :: gdd
    real(real64), intent(inout) :: gdd_cum
    integer(int32), intent(in) :: gdd_method
    real(real64), intent(in) :: t_max
    real(real64), intent(in) :: t_min
    real(real64), intent(in) :: t_base
    real(real64), intent(in) :: t_upp
    integer(int32), intent(in) :: growing_season
    real(real64) :: t_mean
    
    t_mean = mean_temp( &
         gdd_method, &
         t_max, &
         t_min, &
         t_base, &
         t_upp &
         )
    
    gdd = t_mean - t_base
    if ( growing_season == 1 ) then
       gdd_cum = gdd_cum + gdd
    else if ( growing_season == 0 ) then
       gdd_cum = 0.
    end if

  end subroutine update_gdd
  
end module gdd
    

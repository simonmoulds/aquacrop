module temperature_stress
  use types
  implicit none

  real(real64), parameter :: ks_bio_up = 1
  real(real64), parameter :: Ks_bio_lo = 0.02
  real(real64), parameter :: ks_pol_up = 1 
  real(real64), parameter :: ks_pol_lo = 0.001

contains


  
  ! calculate temperature stress effects on biomass production
  !
  ! Input
  !    bio_temp_stress : should we consider temperature stress effects on biomass production
  !    gdd : growing degree days
  !    gdd_up
  !    gdd_lo
  !
  ! Output
  !    kst_bio : stress factor
  !
  ! -------------------------------------------------------------------
  function get_bio_temp_stress( &
       bio_temp_stress, &
       gdd, &
       gdd_up, &
       gdd_lo &
       ) result(kst_bio)
    
    integer(int32), intent(in) :: bio_temp_stress
    real(real64), intent(in) :: gdd
    real(real64), intent(in) :: gdd_up
    real(real64), intent(in) :: gdd_lo
    real(real64) :: kst_bio
    real(real64) :: f_shp_b
    real(real64) :: gdd_rel

    f_shp_b = -1. * log(((ks_bio_lo * ks_bio_up) - 0.98 * ks_bio_lo) / (0.98 * (ks_bio_up - ks_bio_lo)))
    if ( bio_temp_stress == 0 ) then
       kst_bio = 1       
    else
       if ( gdd >= gdd_up ) then
          kst_bio = 1
       else if ( gdd <= gdd_lo ) then
          kst_bio = 0
       else
          gdd_rel = (gdd - gdd_lo) / (gdd_up - gdd_lo)
          kst_bio = (ks_bio_up * ks_bio_lo) / (ks_bio_lo + (ks_bio_up - ks_bio_lo) * exp(-f_shp_b * gdd_rel))
          kst_bio = kst_bio - ks_bio_lo * (1 - gdd_rel)
          
       end if
    end if
    
  end function get_bio_temp_stress

  

  ! calculate effects of heat stress on pollination
  !
  ! TODO
  ! 
  ! -------------------------------------------------------------------
  function get_pol_heat_stress( &
       pol_heat_stress, &
       t_max, &
       t_max_up, &
       t_max_lo, &
       f_shp_b &
       ) result(kst_polh)
    
    integer(int32), intent(in) :: pol_heat_stress
    real(real64), intent(in) :: t_max
    real(real64), intent(in) :: t_max_up
    real(real64), intent(in) :: t_max_lo
    real(real64), intent(in) :: f_shp_b
    real(real64) :: kst_polh
    real(real64) :: t_rel
    
    if ( pol_heat_stress == 0 ) then
       kst_polh = 1
    else
       if ( t_max <= t_max_lo ) then
          kst_polh = 1
       elseif ( t_max >= t_max_up ) then
          kst_polh = 0
       else
          t_rel = (t_max - t_max_lo) / (t_max_up - t_max_lo)
          kst_polh = (ks_pol_up * ks_pol_lo) / (ks_pol_lo + (ks_pol_up - ks_pol_lo) * exp(-f_shp_b * (1 - t_rel)))
       endif
    endif
    
  end function get_pol_heat_stress



  ! calculate effects of cold stress on pollination
  !
  ! TODO
  ! 
  ! -------------------------------------------------------------------
  function get_pol_cold_stress( &
       pol_cold_stress, &
       t_min, &
       t_min_up, &
       t_min_lo, &
       f_shp_b &
       ) result(kst_polc)
    
    integer(int32), intent(in) :: pol_cold_stress
    real(real64), intent(in) :: t_min
    real(real64), intent(in) :: t_min_up
    real(real64), intent(in) :: t_min_lo
    real(real64), intent(in) :: f_shp_b
    real(real64) :: kst_polc
    real(real64) :: t_rel

    if ( pol_cold_stress == 0 ) then
       kst_polc = 1
    else
       if ( t_min >= t_min_up ) then
          kst_polc = 1
       elseif ( t_min <= t_min_lo ) then
          kst_polc = 0
       else
          t_rel = (t_min_up - t_min) / (t_min_up - t_min_lo)
          kst_polc = (ks_pol_up * ks_pol_lo) / (ks_pol_lo + (ks_pol_up - ks_pol_lo) * exp(-f_shp_b * (1 - t_rel)))
          
       endif
    endif
    
  end function get_pol_cold_stress

  

  ! Temperature stress effects
  !
  ! TODO
  !
  ! -------------------------------------------------------------------
  subroutine update_temp_stress(&
       bio_temp_stress, &
       kst_bio, &
       gdd, &
       gdd_up, &
       gdd_lo, &
       pol_heat_stress, &
       kst_polh, &
       t_max, &
       t_max_up, &
       t_max_lo, &
       f_shp_b, &
       pol_cold_stress, &
       kst_polc, &
       t_min, &
       t_min_up, &
       t_min_lo &
       )

    integer(int32), intent(in) :: bio_temp_stress    
    integer(int32), intent(in) :: pol_heat_stress
    integer(int32), intent(in) :: pol_cold_stress
    real(real64), intent(in) :: gdd, gdd_up, gdd_lo
    real(real64), intent(in) :: t_max, t_max_up, t_max_lo
    real(real64), intent(in) :: t_min, t_min_up, t_min_lo
    real(real64), intent(in) :: f_shp_b
    real(real64), intent(inout) :: kst_bio, kst_polh, kst_polc

    kst_bio = get_bio_temp_stress(bio_temp_stress, gdd, gdd_up, gdd_lo)
    kst_polh = get_pol_heat_stress(pol_heat_stress, t_max, t_max_up, t_max_lo, f_shp_b)
    kst_polc = get_pol_cold_stress(pol_cold_stress, t_min, t_min_up, t_min_lo, f_shp_b)
    
  end subroutine update_temp_stress
  
end module temperature_stress

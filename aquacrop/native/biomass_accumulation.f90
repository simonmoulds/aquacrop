module biomass_accumulation
  use types
  use temperature_stress, only: update_temp_stress
  implicit none

contains

  subroutine update_biomass_accum( &
       et_ref, &                ! reference evapotranspiration
       tr, &                    ! transpiration
       tr_pot, &                ! potential transpiration
       b, &                     ! biomass
       b_ns, &                  ! biomass (no water stress)
       bio_temp_stress, &
       gdd, &
       gdd_up, &
       gdd_lo, &
       pol_heat_stress, &
       t_max, &
       t_max_up, &
       t_max_lo, &
       f_shp_b, &
       pol_cold_stress, &
       t_min, &
       t_min_up, &
       t_min_lo, &       
       hi_ref, &
       hi_t, &
       pct_lag_phase, &
       yld_form_cd, &
       wp, &
       wpy, &
       f_co2, &
       crop_type, &
       determinant, &
       growing_season &
       )
    
    real(real64), intent(in) :: et_ref
    real(real64), intent(in) :: tr
    real(real64), intent(in) :: tr_pot
    real(real64), intent(inout) :: b
    real(real64), intent(inout) :: b_ns
    integer(int32), intent(in) :: bio_temp_stress    
    real(real64), intent(in) :: gdd, gdd_up, gdd_lo
    integer(int32), intent(in) :: pol_heat_stress
    real(real64), intent(in) :: t_max, t_max_up, t_max_lo
    integer(int32), intent(in) :: pol_cold_stress
    real(real64), intent(in) :: t_min, t_min_up, t_min_lo
    real(real64), intent(in) :: f_shp_b
    real(real64), intent(in) :: hi_ref
    integer(int32), intent(in) :: hi_t
    real(real64), intent(in) :: pct_lag_phase
    integer(int32), intent(in) :: yld_form_cd
    real(real64), intent(in) :: wp
    real(real64), intent(in) :: wpy
    real(real64), intent(in) :: f_co2
    integer(int32), intent(in) :: crop_type
    integer(int32), intent(in) :: determinant        
    integer(int32), intent(in) :: growing_season

    real(real64) :: kst_bio, kst_polh, kst_polc
    real(real64) :: fswitch
    real(real64) :: wp_adj
    real(real64) :: db
    real(real64) :: db_ns

    if ( growing_season == 1) then

       call update_temp_stress( &
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
       
       ! hi_t = dap - delayed_cds - hi_start_cd - 1

       if ( ( crop_type == 2 .or. crop_type == 3 ) .and. hi_ref > 0 ) then

          ! adjust wp for reproductive stage
          if ( determinant == 1 ) then
             fswitch = pct_lag_phase / 100
          else
             if ( hi_t < ( yld_form_cd / 3 ) ) then
                fswitch = hi_t / ( yld_form_cd / 3 )
             else
                fswitch = 1
             endif
          endif
          wp_adj = wp * (1 - (1 - wpy / 100) * fswitch)
          
       else
          wp_adj = wp
          
       endif

       ! adjust wp for CO2 effects
       wp_adj = wp_adj * f_co2

       ! calculate biomass accumulation on current day
       db_ns = wp_adj * (tr_pot / et_ref) * kst_bio
       db = wp_adj * (tr / et_ref) * kst_bio

       ! TODO: nan check?

       ! update biomass accumulation
       b_ns = b_ns + db_ns
       b = b + db
       
    else
       b_ns = 0
       b = 0
       
    endif
    
  end subroutine update_biomass_accum
  
end module biomass_accumulation

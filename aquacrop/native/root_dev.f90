module root_dev
  use types
  implicit none

contains

  function pot_root_dev( &
       tm_frm_germ, &
       z_min, &
       z_max, &
       fshape_r, &
       pct_z_min, &
       emergence, &
       max_rooting &
       ) result(zr)

    real(real64), intent(in) :: tm_frm_germ
    real(real64), intent(in) :: z_min
    real(real64), intent(in) :: z_max
    real(real64), intent(in) :: fshape_r
    real(real64), intent(in) :: pct_z_min
    real(real64), intent(in) :: emergence
    real(real64), intent(in) :: max_rooting
    real(real64) :: zr
    real(real64) :: z_ini
    real(real64) :: x
    real(real64) :: zr_exp
    real(real64) :: zr_pow
    integer(int32) :: t0
    
    z_ini = z_min * (pct_z_min / 100.)
    t0 = nint(emergence / 2.)       
    if ( tm_frm_germ >= max_rooting ) then
       zr = z_max
    else if ( tm_frm_germ <= t0 ) then
       zr = z_ini
    else
       x = (tm_frm_germ - t0) / (max_rooting - t0)
       zr_exp = 1. / fshape_r
       zr_pow = x ** zr_exp
       zr = z_ini + (z_max - z_ini) * zr_pow
    end if

    if ( zr < z_min ) then
       zr = z_min
    end if
    
  end function pot_root_dev
      
  subroutine update_root_dev( &
       z_root, &
       r_cor, &
       z_min, &
       z_max, &
       pct_z_min, &
       emergence, &
       max_rooting, &
       fshape_r, &
       fshape_ex, &
       sx_bot, &
       sx_top, &
       dap, &
       gdd, &
       gdd_cum, &
       delayed_cds, &
       delayed_gdds, &
       tr_ratio, &
       germination, &
       z_res, &                 ! depth of restrictive soil layer
       water_table, &           ! is groundwater in soil profile?
       z_gw, &                  ! depth of groundwater
       calendar_type, &
       growing_season &         ! is it the growing season?
       )

    real(real64), intent(inout) :: z_root
    real(real64), intent(inout) :: r_cor
    real(real64), intent(in) :: z_min
    real(real64), intent(in) :: z_max
    real(real64), intent(in) :: pct_z_min
    real(real64), intent(in) :: emergence
    real(real64), intent(in) :: max_rooting
    real(real64), intent(in) :: fshape_r
    real(real64), intent(in) :: fshape_ex
    real(real64), intent(in) :: sx_bot
    real(real64), intent(in) :: sx_top
    integer(int32), intent(in) :: dap
    real(real64), intent(in) :: gdd
    real(real64), intent(in) :: gdd_cum
    real(real64), intent(in) :: delayed_cds
    real(real64), intent(in) :: delayed_gdds
    real(real64), intent(in) :: tr_ratio
    integer(int32), intent(in) :: germination
    real(real64), intent(in) :: z_res
    integer(int32), intent(in) :: water_table
    real(real64), intent(in) :: z_gw
    integer(int32), intent(in) :: calendar_type
    integer(int32), intent(in) :: growing_season
    real(real64) :: t_adj
    real(real64) :: t_old
    real(real64) :: zr
    real(real64) :: zr_old
    real(real64) :: dzr
    real(real64) :: f_adj
    
    if ( growing_season == 1 ) then

       ! if today is first day of season, root depth is equal to minimum depth
       if ( dap == 1 ) then
          z_root = z_min
       end if

       ! adjust time for delayed development
       if ( calendar_type == 1 ) then
          t_adj = dap - delayed_cds
       else if ( calendar_type == 2 ) then
          t_adj = gdd_cum - delayed_gdds
       end if

       ! time on previous day
       if ( calendar_type == 1 ) then
          t_old = t_adj - 1
       else if ( calendar_type == 2 ) then
          t_old = t_adj - gdd
       end if

       ! potential root depth on previous day
       zr_old = pot_root_dev( &
            t_old, &
            z_min, &
            z_max, &
            fshape_r, &
            pct_z_min, &
            emergence, &
            max_rooting &
            )
              
       ! potential root depth on current day
       zr = pot_root_dev( &
            t_adj, &
            z_min, &
            z_max, &
            fshape_r, &
            pct_z_min, &
            emergence, &
            max_rooting &
            )
       
       ! determine rate of change
       dzr = zr - zr_old
       
       ! adjust rate of expansion for stomatal water stress
       if ( tr_ratio < 0.9999 ) then
          if ( fshape_ex >= 0 ) then
             dzr = dzr * tr_ratio
          else
             f_adj = (exp(tr_ratio * fshape_ex) - 1) / (exp(fshape_ex) - 1)
             dzr = dzr * f_adj
          end if
       end if

       ! adjust root expansion for failure to germinate
       if ( germination == 0 ) then
          dzr = 0
       end if

       ! new rooting depth
       z_root = z_root + dzr

       ! adjust root depth if restrictive soil layer is present
       if ( z_res > 0 ) then
          if ( z_root > z_res ) then
             r_cor = (2 * (z_root / z_res) * ((sx_top + sx_bot) / 2.) - sx_top) / sx_bot
             z_root = z_res
          end if
       end if

       ! limit rooting depth if groundwater table is present
       if ( water_table == 1 .and. z_gw > 0. ) then
          if ( z_root > z_gw ) then
             z_root = z_gw
             if ( z_root < z_min ) then
                z_root = z_min
             end if
          end if
       end if
    else
       z_root = 0.
    end if
    
  end subroutine update_root_dev
  
end module root_dev

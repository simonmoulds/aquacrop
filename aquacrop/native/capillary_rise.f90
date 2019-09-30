module capillary_rise
  use types
  implicit none
  
contains

  function maximum_cap_rise( &
       k_sat, &
       a_cr, &
       b_cr, &
       zgw, &
       z &
       ) result(max_cr)

    real(real64), intent(in) :: k_sat
    real(real64), intent(in) :: a_cr
    real(real64), intent(in) :: b_cr
    real(real64), intent(in) :: zgw
    real(real64), intent(in) :: z
    real(real64) :: max_cr

    if ( k_sat > 0 .and. zgw > 0 .and. (zgw - z) < 4 ) then
       if ( z > zgw ) then
          max_cr = 99
       else
          max_cr = exp((log(zgw - z) - b_cr) / a_cr)
          max_cr = min(max_cr, 99.)
       end if
    else
       max_cr = 0
    end if
    
  end function maximum_cap_rise
  
    
  subroutine update_cap_rise( &
       cr_tot, &
       th, &
       th_wp, &
       th_fc, &
       th_fc_adj, &
       k_sat, &
       a_cr, &
       b_cr, &
       f_shape_cr, &
       flux_out, &
       water_table, &
       zgw, &
       dz, &
       dz_layer, &
       layer_ix &
       )

    real(real64), intent(inout) :: cr_tot
    real(real64), dimension(:), intent(inout) :: th
    real(real64), dimension(:), intent(in) :: th_wp
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: th_fc_adj
    real(real64), dimension(:), intent(in) :: k_sat
    real(real64), dimension(:), intent(in) :: a_cr
    real(real64), dimension(:), intent(in) :: b_cr
    real(real64), intent(in) :: f_shape_cr
    real(real64), dimension(:), intent(in) :: flux_out
    integer(int32), intent(in) :: water_table
    real(real64), intent(in) :: zgw
    real(real64), dimension(:), intent(in) :: dz
    real(real64), dimension(:), intent(in) :: dz_layer
    integer(int32), dimension(:), intent(in) :: layer_ix
    
    real(real64) :: max_cr, lim_cr
    real(real64) :: df
    real(real64) :: k_rel
    real(real64) :: th_thr
    real(real64) :: dth
    real(real64) :: dth_max
    real(real64) :: zbot, zbotmid, ztoplyr
    real(real64) :: wc_r
    real(real64) :: cr_comp
    integer(int32) :: i, lyri, compi
    integer(int32) :: max_lyr
    integer(int32) :: n_comp, n_layer
    
    n_comp = size(dz, 1)
    n_layer = size(dz_layer, 1)

    if ( water_table == 0 ) then
       cr_tot = 0
       
    else
       zbot = 0
       do i = 1, n_comp
          zbot = zbot + dz(i)
       end do
       zbotmid = zbot - dz(n_comp) / 2
       lyri = layer_ix(n_comp)
       max_cr = maximum_cap_rise(k_sat(lyri), a_cr(lyri), b_cr(lyri), zgw, zbotmid)

       ! find top of next soil layer that is not within the modelled
       ! soil profile. do this by getting the index of the layer
       ! corresponding to the deepest soil compartment; summing
       ! these values will give the distance from the soil surface
       ! to the top of the first layer not included.
       ztoplyr = 0
       max_lyr = layer_ix(n_comp)
       do i = 1, max_lyr
          ztoplyr = ztoplyr + dz_layer(i)
       end do

       ! check for restrictions on upward flow caused by properties
       ! of compartments not modelled in the soil water balance
       lyri = layer_ix(n_comp)
       do while ( ztoplyr < zgw .and. lyri < n_layer )
          lyri = lyri + 1
          lim_cr = maximum_cap_rise(k_sat(lyri), a_cr(lyri), b_cr(lyri), zgw, ztoplyr)
          max_cr = min(max_cr, lim_cr)
          ztoplyr = ztoplyr + dz_layer(lyri)          
       end do

       ! calculate capillary rise
       compi = n_comp
       wc_r = 0
       do while ( nint(max_cr * 1000) > 0 .and. compi > 0 .and. nint(flux_out(compi) * 1000) == 0 )

          ! proceed upwards until maximum capillary rise occurs, soil
          ! surface is reached, or encounter a compartment where
          ! downward drainage/infiltration has already occurred on
          ! current day.
          lyri = layer_ix(compi)
          if ( th(compi) >= th_wp(lyri) .and. f_shape_cr > 0 ) then
             df = 1 - (((th(compi) - th_wp(lyri)) / (th_fc_adj(compi) - th_wp(lyri))) ** f_shape_cr)
             df = min(df, 1.)
             df = max(df, 0.)
          else
             df = 1.
          end if

          ! calculate relative hydraulic conductivity
          th_thr = (th_wp(lyri) + th_fc(lyri)) / 2
          if ( th(compi) < th_thr ) then
             if ( th(compi) <= th_wp(lyri) .or. th_thr <= th_wp(lyri) ) then
                k_rel = 0.
             else
                k_rel = (th(compi) - th_wp(lyri)) / (th_thr - th_wp(lyri))
             end if
          else
             k_rel = 1.
          end if

          ! check if there is space available to store water from
          ! capillary rise
          dth = th_fc_adj(compi) - th(compi)
          dth = nint((dth * 1000)) / 1000

          ! store water if space available
          if ( dth > 0 .and. (zbot - dz(compi) / 2) < zgw ) then
             dth_max = k_rel * df * max_cr / (1000 * dz(compi))
             if ( dth > dth_max ) then
                th(compi) = th(compi) + dth_max
                cr_comp = dth_max * 1000 * dz(compi)
                max_cr = 0
             else
                th(compi) = th_fc_adj(compi)
                cr_comp = dth * 1000 * dz(compi)
                max_cr = (k_rel * max_cr) - cr_comp
             end if
             wc_r = wc_r + cr_comp
          end if

          ! update bottom elevation of compartment
          zbot = zbot - dz(compi)
          compi = compi - 1
          
          ! update restriction on max capillary rise
          if ( compi > 0 ) then
             lyri = layer_ix(compi)
             zbotmid = zbot - (dz(compi) / 2)
             lim_cr = maximum_cap_rise(k_sat(lyri), a_cr(lyri), b_cr(lyri), zgw, zbotmid)
             max_cr = min(max_cr, lim_cr)
          end if
          
       end do
       cr_tot = wc_r
       
    end if
    
  end subroutine update_cap_rise
  
end module capillary_rise

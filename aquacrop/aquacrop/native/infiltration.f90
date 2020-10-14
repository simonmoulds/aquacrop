module infiltration
  use types
  implicit none

contains

  subroutine update_infl( &
       infl, &
       surf_stor, &
       flux_out, &
       deep_perc, &
       runoff, &
       th, &
       irr, &
       app_eff, &
       bund, &
       z_bund, &
       th_sat, &
       th_fc, &
       th_fc_adj, &       
       k_sat, &
       tau, &
       dz, &
       layer_ix &
       )

    real(real64), intent(inout) :: infl 
    real(real64), intent(inout) :: surf_stor
    real(real64), dimension(:), intent(inout) :: flux_out
    real(real64), intent(inout) :: deep_perc
    real(real64), intent(inout) :: runoff
    real(real64), dimension(:), intent(inout) :: th
    real(real64), intent(in) :: irr
    real(real64), intent(in) :: app_eff
    integer(int32), intent(in) :: bund
    real(real64), intent(in) :: z_bund
    real(real64), dimension(:), intent(in) :: th_sat
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: th_fc_adj
    real(real64), dimension(:), intent(in) :: k_sat
    real(real64), dimension(:), intent(in) :: tau
    real(real64), dimension(:), intent(in) :: dz
    integer(int32), dimension(:), intent(in) :: layer_ix

    integer(int32) :: n_comp
    integer(int32) :: i
    integer(int32) :: lyri
    integer(int32) :: precomp
    real(real64), allocatable, dimension(:) :: thnew
    real(real64) :: deep_perc0
    real(real64) :: runoff0
    real(real64) :: infl_tot
    real(real64) :: to_store
    real(real64) :: runoff_ini
    real(real64) :: dthdts
    real(real64) :: dthdt0
    real(real64) :: factor
    real(real64) :: theta0
    real(real64) :: a
    real(real64) :: drainmax
    real(real64) :: drainage
    real(real64) :: diff
    real(real64) :: excess
    
    n_comp = size(dz, 1)
    
    ! make temporary copies of some variables
    allocate(thnew(n_comp))
    thnew = th
    deep_perc0 = deep_perc
    runoff0 = runoff
    
    ! update infiltration rate for irrigation
    infl = infl + (irr * (app_eff / 100.))

    ! determine surface storage (if bunds are present)
    if ( bund == 1 ) then
       if ( z_bund > 0.001 ) then
          infl_tot = infl + surf_stor
          if ( infl_tot > 0. ) then
             if ( infl_tot > k_sat(1) ) then
                ! infiltration is limited by saturated hydraulic
                ! conductivity of surface soil layer
                to_store = k_sat(1)
                ! additional water ponds on surface
                surf_stor = infl_tot - k_sat(1)
             else
                ! all water infiltrates
                to_store = infl_tot
                surf_stor = 0.
             end if
             ! calculate additional runoff
             if ( surf_stor > (z_bund * 1000.) ) then
                ! water overtops bund and runs off
                runoff_ini = surf_stor - (z_bund * 1000.)
                ! surface storage equal to bund height
                surf_stor = z_bund * 1000.
             else
                ! bunds are not overtopped
                runoff_ini = 0.
             end if
          else
             to_store = 0.
             runoff_ini = 0.
          end if
       end if
    else if ( bund == 0 ) then
       ! no bunds on field
       if ( infl > k_sat(1) ) then
          ! infiltration limited by saturated hydraulic
          ! conductivity of top soil layer
          to_store = k_sat(1)
          runoff_ini = infl - k_sat(1)
       else
          ! all water infiltrates
          to_store = infl
          runoff_ini = 0.
       end if
    end if

    ! initialise counters
    i = 0
    runoff = 0

    ! infiltrate incoming water
    if ( to_store > 0 ) then
       do while ( to_store > 0 .and. i < n_comp )
          i = i + 1
          lyri = layer_ix(i)
          ! calculate saturated drainage ability
          dthdts = tau(lyri) * (th_sat(lyri) - th_fc(lyri))
          factor = k_sat(lyri) / (dthdts * 1000. * dz(i))
          ! calculate drainage ability required
          dthdt0 = to_store / (1000. * dz(i))
          ! check drainage ability
          if ( dthdt0 < dthdts ) then
             ! calculate water content, thx, needed to meet drainage dthdt0
             if ( dthdt0 <= 0 ) then
                theta0 = th_fc_adj(i)
             else
                a = 1 + ((dthdt0 * (exp(th_sat(lyri) - th_fc(lyri)) - 1)) / (tau(lyri) * (th_sat(lyri) - th_fc(lyri))))
                theta0 = th_fc(lyri) + log(a)
             end if
             ! limit theta0 to lie between saturation and field capacity
             if ( theta0 > th_sat(lyri) ) then
                theta0 = th_sat(lyri)
             else if ( theta0 <= th_fc_adj(i) ) then
                theta0 = th_fc_adj(i)
                dthdt0 = 0.
             end if
          else
             ! limit water content and drainage to saturation
             theta0 = th_sat(lyri)
             dthdt0 = dthdts
          end if
          
          ! calculate maximum water flow through compartment i
          drainmax = factor * dthdt0 * 1000. * dz(i)
          drainage = drainmax + flux_out(i)
          if ( drainage > k_sat(lyri) ) then
             drainmax = k_sat(lyri) - flux_out(i)
          end if
          
          ! calculate difference between threshold and current water content
          diff = theta0 - th(i)
          if ( diff > 0. ) then
             ! increase water content of compartment i
             thnew(i) = thnew(i) + (to_store / (1000. * dz(i)))
             if ( thnew(i) > theta0 ) then
                ! water remaining that can infiltrate to compartments below
                to_store = (thnew(i) - theta0) * 1000. * dz(i)
                thnew(i) = theta0
             else
                ! all infiltrating water has been stored
                to_store = 0.
             end if
          end if

          ! update outflow from current compartment (drainage + infl)
          flux_out(i) = flux_out(i) + to_store
          
          ! calculate back-up of water into compartments above
          excess = to_store - drainmax
          excess = max(excess, 0.)

          ! update water to store
          to_store = to_store - excess
          
          ! redistribute excess to compartments above
          if ( excess > 0. ) then
             precomp = i + 1
             do while ( excess > 0 .and. precomp /= 1 )
                ! keep storing in compartments above until soil surface is reached
                precomp = precomp - 1
                lyri = layer_ix(precomp)
                flux_out(precomp) = flux_out(precomp) - excess
                thnew(precomp) = thnew(precomp) + (excess / (dz(precomp) * 1000.))
                if ( thnew(precomp) > th_sat(lyri) ) then
                   excess = (thnew(precomp) - th_sat(lyri)) * 1000. * dz(precomp)
                   thnew(precomp) = th_sat(lyri)
                else
                   excess = 0.
                end if
             end do
             
             if ( excess > 0 ) then
                ! any water not stored becomes runoff
                runoff = runoff + excess
             end if
          end if
       end do

       ! infiltration left to store after bottom compartment becomes deep percolation
       deep_perc = to_store
    else
       deep_perc = 0.
       runoff = 0.
    end if

    ! update total runoff
    runoff = runoff + runoff_ini

    ! update surface storage (if bunds are present)
    if ( runoff > runoff_ini ) then
       if ( bund == 1 ) then
          if ( z_bund > 0.001 ) then
             surf_stor = surf_stor + (runoff - runoff_ini)
             if ( surf_stor > (z_bund * 1000) ) then
                runoff = runoff_ini + (surf_stor - (z_bund * 1000.))
                surf_stor = z_bund * 1000.
             else
                runoff = runoff_ini
             end if
          end if
       end if
    end if
    
    th = thnew
    infl = infl - runoff
    deep_perc = deep_perc + deep_perc0
    runoff = runoff + runoff0

    ! clean up
    deallocate(thnew)
    
  end subroutine update_infl
  
end module infiltration


module drainage
  use types
  implicit none

contains

  function compute_dthdt( &
       th, &
       th_sat, &
       th_fc, &
       th_fc_adj, &
       tau &
       ) result(dthdt)

    real(real64), intent(in) :: th
    real(real64), intent(in) :: th_sat
    real(real64), intent(in) :: th_fc
    real(real64), intent(in) :: th_fc_adj
    real(real64), intent(in) :: tau
    real(real64) :: dthdt
    
    if ( th <= th_fc_adj ) then
       dthdt = 0
    else if ( th >= th_sat ) then
       dthdt = tau * (th_sat - th_fc)
       if ( (th - dthdt) < th_fc_adj ) then
          dthdt = th - th_fc_adj
       end if
    else
       dthdt = tau * (th_sat - th_fc) * ((exp(th - th_fc) - 1) / (exp(th_sat - th_fc) - 1))
       if ( (th - dthdt) < th_fc_adj ) then
          dthdt = th - th_fc_adj
       end if
    end if
    
  end function compute_dthdt
      
  subroutine update_drainage( &
       th, &
       deep_perc, &
       flux_out, &
       th_sat, &
       th_fc, &
       k_sat, &
       tau, &
       th_fc_adj, &
       dz, &
       dzsum, &
       layer_ix &
       )

    real(real64), dimension(:), intent(inout) :: th
    real(real64), intent(inout) :: deep_perc
    real(real64), dimension(:), intent(inout) :: flux_out
    real(real64), dimension(:), intent(in) :: th_sat
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: k_sat
    real(real64), dimension(:), intent(in) :: tau
    real(real64), dimension(:), intent(in) :: th_fc_adj
    real(real64), dimension(:), intent(in) :: dz
    real(real64), dimension(:), intent(in) :: dzsum
    integer(int32), dimension(:), intent(in) :: layer_ix

    ! real(real64), allocatable, dimension(:) :: th
    ! real(real64), allocatable, dimension(:) :: fluxout

    real(real64) :: dthdt
    ! real(real64) :: deep_perc
    real(real64) :: draincomp
    real(real64) :: excess
    real(real64) :: prethick
    real(real64) :: drainmax
    real(real64) :: thx
    real(real64) :: a
    integer(int32) :: n_comp
    integer(int32) :: i
    integer(int32) :: lyri
    integer(int32) :: drainability
    integer(int32) :: precomp

    n_comp = size(dz, 1)
    ! allocate(th(n_comp))
    ! allocate(fluxout(n_comp))

    deep_perc = 0.
    ! fluxout = 0.
    
    ! calculate drainage and updated water contents
    do i = 1, n_comp
       ! specify layer for compartment
       lyri = layer_ix(i)
       ! calculate drainage ability of compartment i
       dthdt = compute_dthdt( &
            th(i), &
            th_sat(lyri), &
            th_fc(lyri), &
            th_fc_adj(i), &
            tau(lyri) &
            )

       ! drainage from compartment i (mm)
       draincomp = dthdt * dz(i) * 1000

       ! check drainage ability of compartment i against cumulative
       ! drainage from compartments above
       excess = 0.
       prethick = dzsum(i) - dz(i)
       drainmax = dthdt * 1000 * prethick
       if ( deep_perc <= drainmax ) then
          drainability = 1
       else
          drainability = 0
       end if

       ! drain compartment i
       if ( drainability == 1 ) then
          ! no storage needed; update water content in compartment i
          th(i) = th(i) - dthdt
          ! update cumulative drainage
          deep_perc = deep_perc + draincomp
          ! restrict cumulative drainage to saturated hydraulic
          ! conductivity and adjust excess drainage flow
          if ( deep_perc > k_sat(lyri) ) then
             excess = excess + deep_perc - k_sat(lyri)
             deep_perc = k_sat(lyri)
          end if

       else if ( drainability == 0 ) then
          
          ! storage is needed
          dthdt = deep_perc / (1000 * prethick)
          
          ! calculate value of theta (thx) needed to provide a
          ! drainage ability equal to cumulative drainage
          if ( dthdt <= 0 ) then
             thx = th_fc_adj(i)
          else if ( tau(lyri) > 0 ) then
             a = 1. + ((dthdt * (exp(th_sat(lyri) - th_fc(lyri)) - 1)) / &
                  (tau(lyri) * (th_sat(lyri) - th_fc(lyri))))
             thx = th_fc(lyri) + log(a)
             if ( thx < th_fc_adj(i) ) then
                thx = th_fc_adj(i)
             end if             
             ! thx = max(thx, th_fc_adj(i))
          else
             thx = th_sat(lyri) + 0.01
          end if
          
          ! check thx against hydraulic properties of current soil layer
          if ( thx <= th_sat(lyri) ) then
             ! increase water content in compartment i with cumulative drainage
             th(i) = th(i) + (deep_perc / (1000 * dz(i)))
             ! check updated water content against thx
             if ( th(i) > thx ) then
                ! cumulative drainage is the drainage difference
                ! between thx and new theta plus drainage ability
                deep_perc = (th(i) - thx) * 1000 * dz(i)
                ! calculate drainage ability for thx
                dthdt = compute_dthdt( &
                     thx, &
                     th_sat(lyri), &
                     th_fc(lyri), &
                     th_fc_adj(i), &
                     tau(lyri) &
                     )

                ! update drainage total
                deep_perc = deep_perc + (dthdt * 1000 * dz(i))
                ! restrict cumulative drainage to saturated hydraulic
                ! conductivity and adjust excess drainage flow
                if ( deep_perc > k_sat(lyri) ) then
                   excess = excess + deep_perc - k_sat(lyri)
                   deep_perc = k_sat(lyri)
                end if

                ! update water content
                th(i) = thx - dthdt
             else if ( th(i) > th_fc_adj(i) ) then
                ! calculate drainage ability for updated water content
                dthdt = compute_dthdt( & 
                     th(i), &
                     th_sat(lyri), &
                     th_fc(lyri), &
                     th_fc_adj(i), &
                     tau(lyri) &
                     )
                ! update water content in compartment i
                th(i) = th(i) - dthdt
                ! update cumulative drainage
                deep_perc = dthdt * 1000 * dz(i)
                ! restrict cumulative drainage to saturated hydraulic
                ! conductivity and adjust excess drainage flow
                if ( deep_perc > k_sat(lyri) ) then
                   excess = excess + deep_perc - k_sat(lyri)
                   deep_perc = k_sat(lyri)
                end if

             else
                ! drainage and cumulative drainage are zero as water
                ! content has not risen above field capacity in
                ! compartment i
                deep_perc = 0
             end if

          else if ( thx > th_sat(lyri) ) then
             ! increase water content in compartment i with cumulative
             ! drainage from above
             th(i) = th(i) + (deep_perc / (1000 * dz(i)))
             ! check new water content against hydrualic properties of
             ! soil layer
             if ( th(i) <= th_sat(lyri) ) then
                if ( th(i) > th_fc_adj(i) ) then
                   ! calculate new drainage ability
                   dthdt = compute_dthdt( &
                        th(i), &
                        th_sat(lyri), &
                        th_fc(lyri), &
                        th_fc_adj(i), &
                        tau(lyri) &
                        )

                   ! update water content in compartment i
                   th(i) = th(i) - dthdt
                   ! update cumulative drainage
                   deep_perc = dthdt * 1000 * dz(i)
                   ! restrict cumulative drainage to saturated hydraulic
                   ! conductivity and adjust excess drainage flow
                   if ( deep_perc > k_sat(lyri) ) then
                      excess = excess + deep_perc - k_sat(lyri)
                      deep_perc = k_sat(lyri)
                   end if
                else
                   deep_perc = 0
                end if

             else if ( th(i) > th_sat(lyri) ) then
                ! calculate excess drainage above saturation
                excess = ( th(i) - th_sat(lyri) ) * 1000 * dz(i)

                ! calculate drainage ability for updated water content
                dthdt = compute_dthdt( &
                     th(i), &
                     th_sat(lyri), &
                     th_fc(lyri), &
                     th_fc_adj(i), &
                     tau(lyri) &
                     )

                ! update water content in compartment i
                th(i) = th_sat(lyri) - dthdt
                ! update drainage from compartment i
                draincomp = dthdt * 1000 * dz(i)
                ! update maximum drainage
                drainmax = dthdt * 1000 * prethick
                ! update excess drainage
                if ( drainmax > excess ) then
                   drainmax = excess
                end if                
                ! drainmax = min(drainmax, excess)
                excess = excess - drainmax
                ! update deep_perc and restrict to saturated
                ! hydraulic conductivity of soil layer
                deep_perc = draincomp + drainmax
                if ( deep_perc > k_sat(lyri) ) then
                   excess = excess + deep_perc - k_sat(lyri)
                   deep_perc = k_sat(lyri)
                end if
             end if
          end if
       end if

       flux_out(i) = deep_perc

       ! redistribute excess in compartment above
       if ( excess > 0 ) then
          precomp = i + 1
          do while ( excess > 0 .and. precomp /= 1)
             ! update compartment counter
             precomp = precomp - 1
             ! update layer counter
             lyri = layer_ix(precomp)
             ! update flux from compartment
             if ( precomp < i ) then
                flux_out(precomp) = flux_out(precomp) - excess
             end if
             ! increase water content to store excess
             th(precomp) = th(precomp) + (excess / (1000 * dz(precomp)))
             ! limit water content to saturation and adjust excess counter
             if ( th(precomp) > th_sat(lyri) ) then
                excess = (th(precomp) - th_sat(lyri)) * 1000 * dz(precomp)
                th(precomp) = th_sat(lyri)
             else
                excess = 0
             end if
          end do
       end if
    end do

    ! deep_perc = deep_perc
    ! th = th
    ! Print *, 'th   ', th
    ! Print *, 'th', th
       
  end subroutine update_drainage
  
end module drainage

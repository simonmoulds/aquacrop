module canopy_cover
  use types
  use water_stress, only: update_water_stress
  implicit none

  real(real64), parameter :: zero = 0.0d0
  
contains

  ! Compute canopy development during exponential growth phase
  !
  ! For details see FAO AquaCrop manual, 3.4.2
  !
  ! In:
  !   cc0 :
  !   ccx : maximum fractional canopy cover size (-)
  !   cgc : canopy growth coefficient (day-1/GDD-1)
  !   dt 
  !
  ! Out:
  !   cc  : canopy cover (-)
  !
  ! -------------------------------------------------------------------
  function cc_grow(cc0, ccx, cgc, dt) result(cc)

    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cgc
    real(real64), intent(in) :: dt
    real(real64) :: cc

    ! exponential growth phase
    cc = cc0 * exp(cgc * dt)
    if ( cc > (ccx / 2.) ) then
       ! exponential decline phase
       cc = ccx - 0.25 * (ccx / cc0) * ccx * exp(-cgc * dt)
    end if
    cc = min(cc, ccx)
    cc = max(cc, zero)

  end function cc_grow



  ! Compute canopy decline
  !
  ! In:
  !   ccx : maximum fractional canopy cover size (-)
  !   cdc : canopy decline coefficient (day-1/GDD-1)
  !   dt  : 
  !
  ! Out:
  !   cc  : canopy cover (-)
  !
  ! -------------------------------------------------------------------
  function cc_decl(ccx, cdc, dt) result(cc)

    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cdc
    real(real64), intent(in) :: dt
    real(real64) :: cc
    
    if ( ccx < 0.001 ) then
       cc = 0
    else
       cc = ccx * (1 - 0.05 * (exp(dt * (cdc / ccx)) - 1))
    end if
    
  end function cc_decl
  
  function cc_reqd_time_cgc(cc_prev, cc0, ccx, cgc, dt, tsum) result(treq)    
    real(real64), intent(in) :: cc_prev
    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cgc
    real(real64), intent(in) :: dt
    real(real64), intent(in) :: tsum
    real(real64) :: cgcx
    real(real64) :: treq
    
    if ( cc_prev <= (ccx / 2.) ) then
       cgcx = log(cc_prev / cc0) / (tsum - dt)
    else
       cgcx = log((0.25 * ccx * ccx / cc0) / (ccx - cc_prev)) / (tsum - dt)
    end if
    treq = (tsum - dt) * (cgcx / cgc)
    
  end function cc_reqd_time_cgc
  
  function cc_reqd_time_cdc(cc_prev, ccx, cdc) result(treq)    
    real(real64), intent(in) :: cc_prev
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cdc
    real(real64) :: treq
    
    treq = log(1. + (1. - cc_prev / ccx) / 0.05) / (cdc / ccx)
    
  end function cc_reqd_time_cdc
  
  function adj_ccx(cc_prev, cc0, ccx, cgc, canopy_dev_end, dt, tsum) result(ccx_adj)    
    real(real64), intent(in) :: cc_prev
    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cgc
    real(real64), intent(in) :: canopy_dev_end
    real(real64), intent(in) :: dt
    real(real64), intent(in) :: tsum
    real(real64) :: tmp_tcc
    real(real64) :: ccx_adj
    
    tmp_tcc = cc_reqd_time_cgc(cc_prev, cc0, ccx, cgc, dt, tsum)
    if ( tmp_tcc > 0 ) then
       tmp_tcc = tmp_tcc + (canopy_dev_end - tsum) + dt
       ccx_adj = cc_grow(cc0, ccx, cgc, tmp_tcc)
    else
       ccx_adj = 0.
    end if
    
  end function adj_ccx


  
  subroutine update_ccx_cdc(cdc_adj, ccx_adj, cc_prev, cdc, ccx, dt)    
    real(real64), intent(inout) :: cdc_adj
    real(real64), intent(inout) :: ccx_adj
    real(real64), intent(in) :: cc_prev
    real(real64), intent(in) :: cdc
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: dt
    
    ccx_adj = cc_prev / (1. - 0.05 * (exp(dt * (cdc / ccx)) - 1.))
    cdc_adj = cdc * (ccx_adj / ccx)

  end subroutine update_ccx_cdc

  
  subroutine update_potential_cc( &
       cc_ns, &
       ccx_w_ns, &
       ccx_act_ns, &
       tcc, &
       dtcc, &
       cc0, &
       ccx, &
       cgc, &
       cdc, &
       emergence, &
       maturity, &
       senescence, &
       canopy_dev_end &       
       )

    real(real64), intent(inout) :: cc_ns
    real(real64), intent(inout) :: ccx_w_ns
    real(real64), intent(inout) :: ccx_act_ns
    real(real64), intent(in) :: tcc 
    real(real64), intent(in) :: dtcc 
    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: ccx 
    real(real64), intent(in) :: cgc 
    real(real64), intent(in) :: cdc
    real(real64), intent(in) :: emergence
    real(real64), intent(in) :: maturity
    real(real64), intent(in) :: senescence
    real(real64), intent(in) :: canopy_dev_end

    real(real64) :: tmp_tcc
    
    ! potential canopy development
    if ( tcc < emergence .or. nint(tcc) > maturity ) then
       ! no canopy dev before emergence or after maturity
       cc_ns = 0.

    else if ( tcc < canopy_dev_end ) then
       ! canopy growth can occur
       if ( cc_ns <= cc0 ) then
          cc_ns = cc0 * exp(cgc * dtcc)
       else
          tmp_tcc = tcc - emergence
          cc_ns = cc_grow(cc0, ccx, cgc, tmp_tcc)
       end if
       ! update maximum canopy cover size in growing season
       ccx_act_ns = cc_ns

    else if (tcc > canopy_dev_end) then
       ! no more canopy growth is possible, or canopy in decline
       ccx_w_ns = ccx_act_ns
       if ( tcc < senescence ) then
          cc_ns = cc_ns
          ccx_act_ns = cc_ns
       else
          tmp_tcc = tcc - senescence
          cc_ns = cc_decl(ccx, cdc, tmp_tcc)
       end if
       
    end if
    
  end subroutine update_potential_cc



  subroutine update_actual_cc( &
       cc, &
       ccx_act, &
       crop_dead, &
       cc_prev, &
       tcc, &
       tcc_adj, &
       dtcc, &
       cc0, &
       cc0_adj, &
       ccx, &
       cgc, &
       cdc, &
       emergence, &
       maturity, &
       senescence, &
       canopy_dev_end, &
       ksw_exp &
       )

    real(real64), intent(inout) :: cc
    real(real64), intent(inout) :: ccx_act
    real(real64), intent(inout) :: cc0_adj
    integer(int32), intent(inout) :: crop_dead
    
    real(real64), intent(in) :: cc_prev
    real(real64), intent(in) :: tcc
    real(real64), intent(in) :: tcc_adj
    real(real64), intent(in) :: dtcc 
    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: ccx 
    real(real64), intent(in) :: cgc 
    real(real64), intent(in) :: cdc
    real(real64), intent(in) :: emergence
    real(real64), intent(in) :: maturity
    real(real64), intent(in) :: senescence
    real(real64), intent(in) :: canopy_dev_end
    real(real64), intent(in) :: ksw_exp
    
    real(real64) :: treq, tmp_tcc
    real(real64) :: cgc_adj, cdc_adj, ccx_adj
        
    ! actual canopy development
    if ( tcc_adj < emergence .or. nint(tcc_adj) > maturity ) then
       cc = 0

    else if ( tcc_adj < canopy_dev_end ) then

       if ( cc_prev <= cc0_adj ) then
          cc = cc0_adj * exp(cgc * dtcc)
       else
          if ( cc_prev >= ( 0.9799 * ccx ) ) then
             tmp_tcc = tcc - emergence 
             cc = cc_grow(cc0, ccx, cgc, tmp_tcc)
             cc0_adj = cc0
          else
             cgc_adj = cgc * ksw_exp
             if ( cgc_adj > 0 ) then
                ccx_adj = adj_ccx(cc_prev, cc0_adj, ccx, cgc_adj, canopy_dev_end, dtcc, tcc_adj)
                if ( ccx_adj > 0 ) then
                   if ( abs(cc_prev - ccx) < 0.00001 ) then
                      tmp_tcc = tcc - emergence
                      cc = cc_grow(cc0, ccx, cgc, tmp_tcc)
                   else
                      treq = cc_reqd_time_cgc(cc_prev, cc0_adj, ccx_adj, cgc_adj, dtcc, tcc_adj)
                      tmp_tcc = treq + dtcc
                      if ( tmp_tcc > 0 ) then
                         cc = cc_grow(cc0_adj, ccx_adj, cgc_adj, tmp_tcc)
                      else
                         cc = cc_prev
                      end if
                   end if
                else
                   cc = cc_prev
                end if
             else
                cc = cc_prev
                if ( cc < cc0_adj ) then
                   cc0_adj = cc
                end if
             end if
          end if
       end if

       if ( cc > ccx_act ) then
          ccx_act = cc
       end if

    else if ( tcc_adj > canopy_dev_end ) then

       if ( tcc_adj < senescence ) then
          cc = cc_prev
          if ( cc > ccx_act ) then
             ccx_act = cc
          end if
       else
          cdc_adj = cdc * (ccx_act / ccx)
          tmp_tcc = tcc_adj - senescence
          cc = cc_decl(ccx_act, cdc_adj, tmp_tcc)
       end if

       if ( cc < 0.001 .and. crop_dead == 0 ) then
          cc = 0
          crop_dead = 1
       end if

    end if
    
  end subroutine update_actual_cc

  

  subroutine update_cc_sen_ws( &
       cc, &
       ccx_w, &
       ccx_act, &
       cc0_adj, &
       ccx_early_sen, &       
       t_early_sen, &
       premat_senes, &
       crop_dead, &       
       cc_prev, &
       tcc_adj, &
       dtcc, &
       cc0, &
       ccx, &
       cgc, &
       cdc, &
       emergence, &
       senescence, &
       canopy_dev_end, &
       dr, &
       taw, &
       et_ref, &
       et_adj, &
       p_up1, &
       p_up2, &
       p_up3, &
       p_up4, &
       p_lo1, &
       p_lo2, &
       p_lo3, &
       p_lo4, &
       f_shape_w1, &
       f_shape_w2, &
       f_shape_w3, &
       f_shape_w4 &
       )

    real(real64), intent(inout) :: cc
    real(real64), intent(inout) :: ccx_w
    real(real64), intent(inout) :: ccx_act
    real(real64), intent(inout) :: cc0_adj
    real(real64), intent(inout) :: ccx_early_sen
    real(real64), intent(inout) :: t_early_sen
    integer(int32), intent(inout) :: premat_senes
    integer(int32), intent(inout) :: crop_dead    
    real(real64), intent(in) :: cc_prev
    real(real64), intent(in) :: tcc_adj
    real(real64), intent(in) :: dtcc
    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cgc
    real(real64), intent(in) :: cdc
    real(real64), intent(in) :: emergence
    real(real64), intent(in) :: senescence
    real(real64), intent(in) :: canopy_dev_end    
    real(real64), intent(in) :: dr
    real(real64), intent(in) :: taw
    real(real64), intent(in) :: et_ref
    integer(int32), intent(in) :: et_adj
    real(real64), intent(in) :: p_up1
    real(real64), intent(in) :: p_up2
    real(real64), intent(in) :: p_up3
    real(real64), intent(in) :: p_up4
    real(real64), intent(in) :: p_lo1
    real(real64), intent(in) :: p_lo2
    real(real64), intent(in) :: p_lo3
    real(real64), intent(in) :: p_lo4
    real(real64), intent(in) :: f_shape_w1
    real(real64), intent(in) :: f_shape_w2
    real(real64), intent(in) :: f_shape_w3
    real(real64), intent(in) :: f_shape_w4
    
    real(real64) :: cc_sen
    real(real64) :: cdc_adj, ccx_adj
    real(real64) :: tmp_tcc, treq
    real(real64) :: ksw_exp, ksw_sto, ksw_sen, ksw_pol, ksw_stolin
    integer(int32) :: beta
    
    if ( tcc_adj >= emergence ) then
       if ( tcc_adj < senescence .or. t_early_sen > 0 ) then

          if ( ksw_sen < 1 ) then
             premat_senes = 1

             if ( t_early_sen == 0 ) then
                ccx_early_sen = cc_prev
             end if

             t_early_sen = t_early_sen + dtcc
             beta = 0
             call update_water_stress( &
                  ksw_exp, &
                  ksw_sto, &
                  ksw_sen, &
                  ksw_pol, &
                  ksw_stolin, &
                  dr, &
                  taw, &
                  et_ref, &
                  et_adj, &
                  t_early_sen, &
                  p_up1, &
                  p_up2, &
                  p_up3, &
                  p_up4, &
                  p_lo1, &
                  p_lo2, &
                  p_lo3, &
                  p_lo4, &
                  f_shape_w1, &
                  f_shape_w2, &
                  f_shape_w3, &
                  f_shape_w4, &
                  beta)

             if ( ksw_sen > 0.99999 ) then
                cdc_adj = 0.0001
             else
                cdc_adj = (1 - (ksw_sen ** 8)) * cdc
             end if

             if ( ccx_early_sen < 0.001 ) then
                cc_sen = 0
             else
                treq = cc_reqd_time_cdc(cc_prev, ccx_early_sen, cdc_adj)
                tmp_tcc = treq + dtcc
                cc_sen = cc_decl(ccx_early_sen, cdc_adj, tmp_tcc)
             end if

             if ( tcc_adj < senescence ) then
                
                if ( cc_sen > ccx ) then
                   cc_sen = ccx
                end if
                cc = cc_sen
                
                if ( cc > cc_prev ) then
                   cc = cc_prev
                end if
                ccx_act = cc
                
                if ( cc < cc0 ) then
                   cc0_adj = cc
                else
                   cc0_adj = cc0
                end if
                
             else
                if ( cc_sen < cc ) then
                   cc = cc_sen
                end if
                
             end if

             if ( cc < 0.001 .and. crop_dead == 0 ) then
                cc = 0
                crop_dead = 1
             end if

          else
             premat_senes = 0
             if ( tcc_adj > senescence .and. t_early_sen > 0 ) then
                tmp_tcc = tcc_adj - dtcc - senescence
                call update_ccx_cdc( &
                     cdc_adj, &
                     ccx_adj, &
                     cc_prev, &
                     cdc, &
                     ccx, &
                     tmp_tcc &
                     )
                
                tmp_tcc = tcc_adj - senescence
                cc = cc_decl(ccx_adj, cdc_adj, tmp_tcc)
                if ( cc < 0.001 .and. crop_dead == 0 ) then
                   cc = 0
                   crop_dead = 1
                end if
                
             end if
             t_early_sen = 0
             
          end if
          ccx_w = max(ccx_w, cc)

       end if
    end if
    
  end subroutine update_cc_sen_ws



  subroutine update_canopy_cover( &
       cc, &
       cc_prev, &
       cc_adj, &
       cc_ns, &
       cc_adj_ns, &
       ccx_w, &
       ccx_act, &
       ccx_w_ns, &
       ccx_act_ns, &       
       cc0_adj, &
       ccx_early_sen, &       
       t_early_sen, &
       premat_senes, &
       crop_dead, &       
       gdd, &
       gddcum, &       
       cc0, &
       ccx, &
       cgc, &
       cdc, &
       emergence, &
       maturity, &
       senescence, &
       canopy_dev_end, &
       dr, &
       taw, &
       et_ref, &
       et_adj, &
       p_up1, &
       p_up2, &
       p_up3, &
       p_up4, &
       p_lo1, &
       p_lo2, &
       p_lo3, &
       p_lo4, &
       f_shape_w1, &
       f_shape_w2, &
       f_shape_w3, &
       f_shape_w4, &
       growing_season, &
       calendar_type, &
       dap, &
       delayed_cds, &
       delayed_gdds &
       )

    real(real64), intent(inout) :: cc
    real(real64), intent(inout) :: cc_prev
    real(real64), intent(inout) :: cc_adj
    real(real64), intent(inout) :: cc_ns
    real(real64), intent(inout) :: cc_adj_ns
    real(real64), intent(inout) :: ccx_w
    real(real64), intent(inout) :: ccx_act
    real(real64), intent(inout) :: ccx_w_ns
    real(real64), intent(inout) :: ccx_act_ns
    real(real64), intent(inout) :: cc0_adj
    real(real64), intent(inout) :: ccx_early_sen
    real(real64), intent(inout) :: t_early_sen
    integer(int32), intent(inout) :: premat_senes
    integer(int32), intent(inout) :: crop_dead
    
    real(real64), intent(in) :: gdd
    real(real64), intent(in) :: gddcum
    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cgc
    real(real64), intent(in) :: cdc
    real(real64), intent(in) :: emergence
    real(real64), intent(in) :: maturity
    real(real64), intent(in) :: senescence
    real(real64), intent(in) :: canopy_dev_end
    
    real(real64), intent(in) :: dr
    real(real64), intent(in) :: taw
    real(real64), intent(in) :: et_ref
    integer(int32), intent(in) :: et_adj
    real(real64), intent(in) :: p_up1
    real(real64), intent(in) :: p_up2
    real(real64), intent(in) :: p_up3
    real(real64), intent(in) :: p_up4
    real(real64), intent(in) :: p_lo1
    real(real64), intent(in) :: p_lo2
    real(real64), intent(in) :: p_lo3
    real(real64), intent(in) :: p_lo4
    real(real64), intent(in) :: f_shape_w1
    real(real64), intent(in) :: f_shape_w2
    real(real64), intent(in) :: f_shape_w3
    real(real64), intent(in) :: f_shape_w4    
    integer(int32), intent(in) :: growing_season
    integer(int32), intent(in) :: calendar_type
    integer(int32), intent(in) :: dap
    integer(int32), intent(in) :: delayed_cds
    real(real64), intent(in) :: delayed_gdds
    
    ! real(real64) :: cc_prev
    real(real64) :: cc_sen
    real(real64) :: cgc_adj, cdc_adj, ccx_adj
    real(real64) :: tcc, tmp_tcc, dtcc, tcc_adj, treq
    real(real64) :: ksw_exp, ksw_sto, ksw_sen, ksw_pol, ksw_stolin
    integer(int32) :: beta

    ! update cc_prev (cc from previous timestep)
    cc_prev = cc
    
    if ( growing_season == 1 ) then

       beta = 1
       call update_water_stress( &
            ksw_exp, &
            ksw_sto, &
            ksw_sen, &
            ksw_pol, &
            ksw_stolin, &
            dr, &
            taw, &
            et_ref, &
            et_adj, &
            t_early_sen, &
            p_up1, &
            p_up2, &
            p_up3, &
            p_up4, &
            p_lo1, &
            p_lo2, &
            p_lo3, &
            p_lo4, &
            f_shape_w1, &
            f_shape_w2, &
            f_shape_w3, &
            f_shape_w4, &
            beta)
       
       if ( calendar_type == 1 ) then
          tcc = dap
          dtcc = 1.
          tcc_adj = dap - delayed_cds
       else
          tcc = gddcum
          dtcc = gdd
          tcc_adj = gddcum - delayed_gdds
       end if

       call update_potential_cc( &
            cc_ns, &
            ccx_w_ns, &
            ccx_act_ns, &
            tcc, &
            dtcc, &
            cc0, &
            ccx, &
            cgc, &
            cdc, &
            emergence, &
            maturity, &
            senescence, &
            canopy_dev_end &
            )

       call update_actual_cc( &
            cc, &
            ccx_act, &
            crop_dead, &
            cc_prev, &
            tcc, &
            tcc_adj, &
            dtcc, &
            cc0, &
            cc0_adj, &
            ccx, &
            cgc, &
            cdc, &
            emergence, &
            maturity, &
            senescence, &
            canopy_dev_end, &
            ksw_exp &
            )       

       ! adjust for canopy senescence due to water stress
       call update_cc_sen_ws( &
            cc, &
            ccx_w, &
            ccx_act, &
            cc0_adj, &
            ccx_early_sen, &       
            t_early_sen, &
            premat_senes, &
            crop_dead, &       
            cc_prev, &
            tcc_adj, &
            dtcc, &
            cc0, &
            ccx, &
            cgc, &
            cdc, &
            emergence, &
            senescence, &
            canopy_dev_end, &
            dr, &
            taw, &
            et_ref, &
            et_adj, &
            p_up1, &
            p_up2, &
            p_up3, &
            p_up4, &
            p_lo1, &
            p_lo2, &
            p_lo3, &
            p_lo4, &
            f_shape_w1, &
            f_shape_w2, &
            f_shape_w3, &
            f_shape_w4 &
            )

       ! ensure potential cc is not lower than actual
       if ( cc_ns < cc ) then
          cc_ns = cc
          if ( tcc < canopy_dev_end ) then
             ccx_act_ns = cc_ns
          end if
       end if
       
       ! adjust for microadvective effects
       cc_adj = (1.72  * cc) - (cc ** 2) + (0.3 * cc ** 3)
       cc_adj_ns = (1.72 * cc_ns) - (cc_ns ** 2) + (0.3 * cc_ns ** 3)
       
    else
       cc = 0
       cc_adj = 0
       cc_ns = 0
       cc_adj_ns = 0
       ccx_w = 0
       ccx_act = 0
       ccx_w_ns = 0
       ccx_act_ns = 0
       
    end if
    
  end subroutine update_canopy_cover
  
end module canopy_cover

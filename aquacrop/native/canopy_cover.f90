module canopy_cover
  use types
  implicit none

  ! real(real64), parameter :: zero = 0.0d0
  ! integer(int32), parameter :: one = 1
  ! integer(int32), parameter :: two = 2
  ! integer(int32), parameter :: four = 4
  
contains

  function cc_dev(cc0, ccx, cgc, dt) result(cc)

    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cgc
    real(real64), intent(in) :: cdc
    integer(int32), intent(in) :: dt
    real(real64) :: cc

    cc = cc0 * exp(cgc * dt)
    if ( cc > (ccx / 2) ) then       
       cc = ccx - 0.25 * (ccx / cc0) * ccx * exp(-cgc * dt)
    end if
    cc = min(cc, ccx)
    cc = max(cc, 0)

  end function cc_dev

  function cc_decl(ccx, cdc, dt) result(cc)

    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cdc
    integer(int32), intent(in) :: dt
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
    real(real64), intent(in) :: cdc
    real(real64), intent(in) :: dt
    real(real64), intent(in) :: tsum ! real or int?
    real(real64) :: treq
    
    if ( cc_prev <= (ccx / 2) ) then
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
    
    treq = log(1 + (1 - cc_prev / ccx) / 0.05) / (cdc / ccx)
    
  end function cc_reqd_time_cdc


  
  function adj_ccx(cc_prev, cc0, ccx, cgc, canopy_dev_end, dt, tsum) result(ccx_adj)
    
    real(real64), intent(in) :: cc_prev
    real(real64), intent(in) :: cc0
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: cgc
    real(real64), intent(in) :: canopy_dev_end
    integer(int32), intent(in) :: dt
    integer(int32), intent(in) :: tsum
    real(real64) :: ccx_adj
    
    tcc_tmp = cc_reqd_time_cgc(cc_prev, cc0, ccx, cgc, dt, tsum)
    if ( tcc_tmp > 0 ) then
       tcc_tmp = tcc_tmp + (canopy_dev_end - tsum) + dt
       ccx_adj = cc_dev(cc0, ccx, cgc, dt)
    else
       ccx_adj = 0
    end if
    
  end function adj_ccx
  
! %% Get time required to reach CC on previous day %%
! tCCtmp = AOS_CCRequiredTime(CCprev,CCo,CCx,CGC,CDC,dt,tSum,'CGC');

! %% Determine CCx adjusted %%
! if tCCtmp > 0
!     tCCtmp = tCCtmp+(Crop.CanopyDevEnd-tSum)+dt;
!     CCxAdj = AOS_CCDevelopment(CCo,CCx,CGC,CDC,tCCtmp,'Growth');
! else
!     CCxAdj = 0;
! end
  
  subroutine canopy_cover( &
       cc, cc_ns, &
       gdd, gddcum, &
       emergence, maturity, &
       canopy_dev_end, &
       growing_season, senescence, premat_senes, &
       calendar_type, dap, delayed_cds, delayed_gdds &
       )

    real(real64), intent(inout) :: cc, cc_ns, ccx_act_ns, ccx_act, ccx_w_ns, cc0_adj, ccx_early_sen
    integer(real64), intent(inout) :: crop_dead, t_early_sen, premat_senes
    
    integer(int32), intent(in) :: gdd
    integer(int32), intent(in) :: gddcum
    integer(int32), intent(in) :: emergence
    integer(int32), intent(in) :: maturity
    integer(int32), intent(in) :: senescence
    integer(int32), intent(in) :: canopy_dev_end
    integer(int32), intent(in) :: growing_season
    integer(int32), intent(in) :: calendar_type
    integer(int32), intent(in) :: dap
    integer(int32), intent(in) :: delayed_cds
    integer(int32), intent(in) :: delayed_gdds
    real(real64) :: ksw_exp, ksw_sen
    real(real64) :: cgc_adj, ccx_adj
    real(real64) :: tcc, dtcc, tccadj

    ! TODO: root_zone_water
    ! TODO: water_stress - this will update ksw_exp etc. 

    ! make a copy of cc before any changes are made
    cc_prev = cc
    cc_ns_prev = cc_ns
    
    if ( growing_season == 1 ) then
       
       if ( calendar_type == 1 ) then
          tcc = dap
          dtcc = 1
          tccadj = dap - delayed_cds
       else
          tcc = gddcum
          dtcc = gdd
          tccadj = gddcum - delayed_gdds
       end if

       ! potential canopy development
       ! TODO: round (nint?)
       if ( tcc < emergence .or. round(tcc) > maturity ) then
          cc_ns = 0
          
       else if ( tcc < canopy_dev_end ) then
          if ( cc_ns_prev <= cc0 ) then
             cc_ns = cc0 * exp(cgc * dtcc)
          else
             cc_ns = cc_dev(cc0, ccx, cgc, dt)
          end if
          ccx_act_ns = cc_ns
          
       else if (tcc > canopy_dev_end)
          ccx_w_ns = ccx_act_ns
          if ( tcc < senescence ) then
             cc_ns = cc_ns_prev
             ccx_act_ns = cc_ns
          else
             tmp_tcc = tcc - senescence
             cc_ns = cc_decl(ccx, cdc, tmp_tcc)
          end if
       end if
       
       ! actual canopy development
       if ( tccadj < emergence .or. round(tccadj) > maturity ) then
          cc = 0

       else if ( tccadj < canopy_dev_end ) then

          if ( cc_prev <= cc0_adj ) then
             cc = cc0_adj * exp(cgc * dtcc)
          else
             if ( cc_prev >= 0.9799 * ccx ) then
                tmp_tcc = tcc - emergence 
                cc = cc_dev(cc0, ccx, cgc, tmp_tcc)
                cc0_adj = cc0
             else
                cgc_adj = cgc * ksw_exp
                if ( cgc_adj > 0 ) then
                   ccx_adj = adj_ccx(cc_prev, cc0_adj, ccx, cgc_adj, cdc, dtcc, tcc_adj)
                   if ( ccx_adj > 0 ) then
                      if ( abs(cc_prev - ccx) < 0.00001 ) then !TODO: check abs is doing what you think
                         tmp_tcc = tcc - emergence
                         cc = cc_dev(cc0, ccx, cgc, tmp_tcc)
                      else
                         treq = cc_reqd_time_cgc()
                         tmp_tcc = treq + dtcc
                         if ( tmp_tcc > 0 ) then
                            cc = cc_dev(cc0_adj, ccx_adj, cgc_adj, tmp_tcc)
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

       if ( tcc_adj >= emergence ) then
          if ( tcc_adj < senescence .or. t_early_sen > 0 ) then
             if ( ksw_sen < 1 ) then
                premat_senes = 1
                if ( t_early_sen == 0 ) then
                   ccx_early_sen = cc_prev
                end if
                t_early_sen = t_early_sen + dtcc
                ! TODO: update water stress
                ! beta = false;
                ! Ksw = AOS_WaterStress(Crop,NewCond,Dr,TAW,Et0,beta);
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
                   cc_sen = min(cc_sen, ccx)
                   cc = min(cc, cc_prev)
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
                   ! TODO
                   ! [CCXadj,CDCadj] = AOS_UpdateCCxCDC(InitCond.CC,...
                   !     Crop.CDC,Crop.CCx,tmp_tCC);
                   tmp_tcc = tcc_adj - senescence
                   cc = cc_decl(ccx_adj, cdc_adj, tmp_tcc)
                   if ( cc < 0.001 .and. crop_dead == 0 ) then
                      cc = 0
                      crop_dead = 1
                   end if
                end if
                t_early_sen = 0
                
             end if

             if ( cc > ccx_w ) then
                ccx_w = cc
             end if
          end if
       end if
       
       if ( cc_ns < cc ) then
          cc_ns = cc
          if ( tcc < canopy_dev_end ) then
             ccx_act_ns = cc_ns
          end if
       end if
       cc_adj = (1.72  * cc) - (cc ** 2) + (0.3 * cc ** 3)
       cc_adj_ns = (1.72 * cc_ns) - (cc_ns ** 2) + (0.3 * cc_ns ** 3)
    else
       cc = 0
       cc_adj = 0
       cc_ns = 0
       cc_adj_ns = 0
       ccxw = 0
       ccx_act = 0
       ccxw_ns = 0
       ccx_act_ns = 0
    end if
  end subroutine canopy_cover
  
end module canopy_cover

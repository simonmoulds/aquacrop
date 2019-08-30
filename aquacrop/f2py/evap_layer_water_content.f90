module soil_evaporation
  implicit none

  ! https://www.fortran90.org/src/best-practices.html
  
contains


  
  ! Get the index of the deepest compartment in evaporation layer.
  !
  ! Input:
  !    evap_z : depth of soil evaporation layer (m)
  !    dz_sum : accumulated compartment depths (m)
  !    n_comp : number of compartments (m)
  ! 
  ! Output:
  !    n_comp_max : index of deepest compartment
  ! 
  ! -------------------------------------------------------------------
  function get_max_comp_idx(evap_z, dz_sum, n_comp) result (n_comp_max)
    
    real(8), dimension(n_comp), intent(in) :: dz_sum
    real(8), intent(in) :: evap_z
    integer, intent(in) :: n_comp
    integer, intent(out) :: n_comp_max

    n_comp_max = 1
    do while (dz_sum(n_comp_max) < evap_z)
       n_comp_max = n_comp_max + 1
    end do
    
  end function get_max_comp_idx



  ! Compute water contents in soil evaporation layer.
  !
  ! Input:
  !    th, th_sat, th_fc, th_wilt th_dry : volumetric water contents (m3 m-3)
  !    evap_z : depth of soil evaporation layer (m)
  !    dz     : thickness of each soil compartment (m)
  !    dz_sum : accumulated compartment depth (m)
  !    n_comp : number of compartments
  !
  ! Output:
  !    w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp : depth of water in soil evaporation layer (mm)
  !
  ! -------------------------------------------------------------------
  subroutine get_evap_lyr_wc(th, th_sat, th_fc, th_wilt, th_dry, &
       w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry, &
       evap_z, dz, dz_sum, n_comp)
  
    integer, intent(in) :: n_comp
    real(8), dimension(n_comp), intent(in) :: th, th_sat, th_fc, th_wilt, th_dry, dz, dz_sum
    real(8), intent(in) :: evap_z
    real(8), intent(out) :: w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry
    integer :: i, n_comp_max
    real(8) :: factor

    ! initialize water content to zero
    w_evap_act = 0
    w_evap_sat = 0
    w_evap_fc = 0
    w_evap_wp = 0
    w_evap_dry = 0

    ! get the index of the deepest compartment in the soil evaporation layer
    max_comp_idx = get_max_comp_idx(dz_sum, evap_z, n_comp)

    ! loop through compartments, calculating the depth of water in each
    do i = 1, max_comp_idx       
       if (dz_sum(i) > evap_z) then
          factor = 1 - (dz_sum(i) - evap_z) / dz(i)
       else
          factor = 1
       end if       
       w_evap_act = w_evap_act + factor * 1000 * th(i) * dz(i)
       w_evap_sat = w_evap_sat + factor * 1000 * th_sat(i) * dz(i)
       w_evap_fc = w_evap_fc + factor * 1000 * th_fc(i) * dz(i)
       w_evap_wp = w_evap_wp + factor * 1000 * th_wilt(i) * dz(i)
       w_evap_dry = w_evap_dry + factor * 1000 * th_dry(i) * dz(i)
    end do
    
  end subroutine get_evap_lyr_wc



  ! ! Prepare stage two soil evaporation.
  ! !
  ! ! Input:
  ! !    th, th_sat, th_fc, th_wilt th_dry : volumetric water contents (m3 m-3)
  ! !    evap_z : depth of soil evaporation layer (m)
  ! !    dz     : thickness of each soil compartment (m)
  ! !    dz_sum : accumulated compartment depth (m)
  ! !    n_comp : number of compartments
  ! !
  ! ! Output:
  ! !    w_stage_two
  ! !
  ! ! -------------------------------------------------------------------
  ! function prepare_stage_two_evaporation(th, th_sat, th_fc, th_wilt, &
  !      th_dry, w_surf, evap_z, dz, dz_sum, n_comp) result (w_stage_two)
    
  !   integer, intent(in) :: n_comp
  !   real(8), dimension(n_comp), intent(in) :: th, th_sat, th_fc, th_wilt, th_dry, dz, dz_sum
  !   real(8), intent(in) :: evap_z
  !   real(8), intent(out) :: w_stage_two
  !   real(8) :: w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry

  !   ! get water contents in soil evaporation layer
  !   call get_evap_lyr_wc(th, th_sat, th_fc, th_wilt, th_dry, &
  !      w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry, &
  !      evap_z, dz, dz_sum, n_comp)

  !   w_stage_two = (w_evap_act - (w_evap_fc - rew)) / (w_evap_sat - (w_evap_fc - rew))    
  !   if (w_stage_two < 0) then
  !      w_stage_two = 0
  !   end if
  ! end function prepare_stage_two_evaporation



  ! ! Prepare stage one soil evaporation.
  ! !
  ! ! Input
  ! !    prec
  ! !    irr
  ! !    irr_method
  ! !    evap_z_min
  ! !    infl
  ! !    rew
  ! !    w_stage_two
  ! !    w_surf
  ! !    evap_z
  ! ! 
  ! ! Output:
  ! !    w_stage_two
  ! !    w_surf
  ! !    evap_z
  ! !
  ! ! -------------------------------------------------------------------
  ! subroutine prepare_soil_evaporation_stage_one(prec, irr, irr_method &
  !      evap_z_min, infl, rew, w_stage_two, w_surf, evap_z)
    
  !   real(8), intent(in) :: prec, irr
  !   integer, intent(in) :: irr_method, evap_z_min
  !   real(8), intent(inout) :: w_stage_two, w_surf, evap_z
    
  !   if (prec > 0 .or. (irr > 0 .and. .not. irr_method == 4)) then
  !      if (infl > 0) then
  !         w_surf = infl
  !         if (w_surf > rew) then
  !            w_surf = rew
  !         end if
  !         w_stage_two = 0
  !         evap_z = evap_z_min
  !      end if
  !   end if
  ! end subroutine prepare_soil_evaporation_stage_one

  function potential_soil_evaporation_rate(et_ref, cc, cc_adj, &
       ccx_act, growing_season, premat_senes, kex, ccxw, fwcc, &
       t_adj, senescence) result (es_pot)
    
    integer, intent(in) :: t_adj, senescence, growing_season, premat_senes
    real(8), intent(in) :: et_ref, cc, cc_adj, ccx_act, kex, ccxw, fwcc 
    real(8) :: es_pot_min, es_pot_max, mult, ccx_act_adj
    real(8) :: es_pot
    
    if (growing_season == 1) then
       es_pot_max = kex * et_ref * (1 - ccxw * (fwcc / 100))
       es_pot = kex * (1 - cc_adj) * et_ref
       if (adjusted_time > senescence .and. ccx_act > 0) then
          if (cc > (ccx_act / 2)) then
             if (cc > ccx_act) then
                mult = 0
             else
                mult = (ccx_act - cc) / (ccx_act / 2)
             end if
          else
             mult = 1
          end if
          es_pot = es_pot * (1 - ccx_act * (fwcc / 100) * mult)
          ccx_act_adj = (1.72 * ccx_act) + (ccx_act ** 2) - 0.3 * (ccx_act ** 3)
          es_pot_min = kex * (1 - ccx_act_adj) * et_ref
          if (es_pot_min < 0) then
             es_pot_min = 0
          end if
          if (es_pot < es_pot_min) then
             es_pot = es_pot_min
          else if es_pot > es_pot_max
             es_pot = es_pot_max
          end if
       end if
       if (premat_senes == 1) then
          if (es_pot > es_pot_max) then
             es_pot = es_pot_max
          end if          
       end if
    else
       es_pot = kex * et_ref
    end if
    
  end function potential_soil_evaporation_rate

  function potential_soil_evaporation_rate_with_mulches(mulches, growing_season, surface_storage, f_mulch, mulch_pct_gs, mulch_pct_os, es_pot) result (es_pot_mulch)
    
    integer, intent(in) :: mulches, growing_season
    real(8), intent(in) :: surface_storage, f_mulch, mulch_pct_gs, mulch_pct_os
    real(8), intent(inout) :: es_pot
    real(8) :: es_pot_mulch
    
    if (surface_storage < 0.000001) then
       if (mulches == 0) then
          es_pot_mul = es_pot
       else if (mulches == 1) then
          if (growing_season == 1) then
             es_pot_mul = es_pot * (1 - f_mulch * (mulch_pct_gs / 100))
          else if (growing_season == 0) then
             es_pot_mul = es_pot * (1 - f_mulch * (mulch_pct_os / 100))
          end if
       end if
    else
       es_pot_mul = es_pot
    end if    
  end function potential_soil_evaporation_rate_with_mulches

  function potential_soil_evaporation_rate_with_irrigation(prec, irr, irr_method, surface_storage, wet_surf) result (es_pot_irr)
    integer, intent(in) :: irr_method
    real(8), intent(in) :: prec, irr, surface_storage, wet_surf
    real(8) :: es_pot_irr
    if (irr > 0 .and. .not. irr_method == 4) then
       if (rain > 1 .or. surface_storage > 0) then
          es_pot_irr = es_pot
       else
          es_pot_irr = es_pot * (wet_surf / 100)
       end if
    else
       es_pot_irr = es_pot
    end if
  end function potential_soil_evaporation_rate_with_irrigation

  subroutine surface_evaporation(es_pot, es_act, surface_storage, rew, w_stage_two, w_surf, evap_z)
    real(8), intent(in) :: es_pot, rew
    real(8), intent(inout) :: es_act, surface_storage, w_surf, w_stage_two, evap_z
    if (surface_storage > 0) then
       if (surface_storage > es_pot) then
          es_act = es_pot
          surface_storage = surface_storage - es_act
       else
          es_act = surface_storage
          surface_storage = 0
          w_surf = rew
          w_stage_two = 0
          evap_z = evap_z_min
       end if
    end if
  end subroutine surface_evaporation

  subroutine extract_water(to_extract, to_extract_stage, th, th_dry, dz_sum, evap_z, n_comp)

    integer, intent(in) :: n_comp
    real(8), intent(in) :: evap_z
    real(8), intent(inout) :: to_extract, to_extract_stage
    real(8), dimension(n_comp), intent(inout) :: th
    real(8), dimension(n_comp), intent(in) th_dry, dz, dz_sum
    
    if (extract_pot_stage_one > 0) then
       n_comp_max = get_max_comp_idx(dz_sum, evap_z, n_comp)
       comp = 0
       do while (extract_pot_stage_one > 0 .and. comp < n_comp_max)
          comp = comp + 1
          if (dz_sum(comp) > evap_z_min) then
             factor = 1 - (dz_sum(comp) - evap_z) / dz(comp)
          else
             factor = 1
          end if
          w_dry = 1000 * th_dry(comp) * dz(comp)
          w = 1000 * th(comp) * dz(comp)
          av_w = (w - w_dry) * factor
          if (av_w < 0) then
             av_w = 0
          end if
          if (av_w >= extract_pot_stage_one) then
             es_act = es_act + extract_pot_stage_one
             w = w - extract_pot_stage_one
             to_extract = to_extract - extract_pot_stage_one
             extract_pot_stage_one = 0
          else
             es_act = es_act + av_w
             extract_pot_stage_one = extract_pot_stage_one - av_w
             to_extract = to_extract - av_w
             w = w - av_w
          end if
          th(comp) = w / (1000 * dz(comp))
       end do

       ! ! update surface evaporation layer water balance
       ! w_surf = w_surf - es_act
       ! if (w_surf < 0 .or. extract_pot_stage_one > 0.0001) then
       !    w_surf = 0
       ! end if
    end if
  end subroutine extract_water
  
  subroutine stage_one_evaporation(to_extract, w_surf, es_act, w_surf, evap_z, evap_z_min, th, dz, dz_sum, n_comp)
    real(8), intent(in) :: es_act
    real(8), dimension(n_comp), intent(in) :: dz, dz_sum, th_dry
    real(8), intent(inout) :: to_extract, w_surf
    real(8), dimension(n_comp), intent(inout) :: th    
    real(8) :: extract_pot_stage_one, factor
    integer :: comp, n_comp_max

    ! extract water
    ! to_extract = es_pot - es_act
    to_extract_stage_one = min(to_extract, w_surf)
    call extract_water(to_extract, to_extract_stage_one, th, th_dry, dz_sum, evap_z, n_comp)
    
    ! update surface evaporation layer water balance
    if (extract_pot_stage_one > 0) then
       w_surf = w_surf - es_act
       if (w_surf < 0 .or. extract_pot_stage_one > 0.0001) then
          w_surf = 0
       end if
    end if    
  end subroutine stage_one_evaporation

  subroutine relative_depletion(th, th_sat, th_fc, th_wilt, th_dry, &
       w_upper, w_lower, w_rel, rew, evap_z, dz, dz_sum, n_comp)
    
    integer, intent(in) :: n_comp
    real(8), dimension(n_comp), intent(in) :: th, th_sat, th_fc, th_wilt, th_dry, dz, dz_sum
    real(8), intent(in) :: rew, evap_z
    real(8), intent(out) :: w_upper, w_lower, w_rel
    real(8) :: w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry
    
    call get_evap_lyr_wc(th, th_sat, th_fc, th_wilt, &
       th_dry, evap_z, dz, dz_sum, w_evap_act, w_evap_sat, &
       w_evap_fc, w_evap_wp, w_evap_dry, n_comp)
    w_upper = w_stage_two * (w_evap_sat - (w_evap_fc - rew)) + (w_evap_fc - rew)
    w_lower = w_evap_dry
    w_rel = (w_evap_act - w_lower) / (w_upper - w_lower)    
  end subroutine relative_depletion
    
  subroutine stage_two_evaporation(to_extract, evap_time_steps, th, th_sat, th_fc, th_wilt, th_dry, evap_z, dz, dz_sum, rew, evap_z, evap_z_max, evap_z_min, fwrel_exp)
    real(8), intent(inout) :: to_extract
    integer, intent(in) :: evap_time_steps
    real(8) :: w_upper, w_lower, w_rel

    if (to_extract > 0) then
       edt = to_extract / evap_time_steps
       do i, evap_time_steps
          call relative_depletion(th, th_sat, th_fc, th_wilt, th_dry, &
               w_upper, w_lower, w_rel, rew, evap_z, dz, dz_sum, n_comp)

          if (evap_z_max > evap_z_min) then
             w_check = f_wrel_exp * ((evap_z_max - evap_z) / (evap_z_max - evap_z_min))
             do while (w_rel < w_check .and. evap_z < evap_z_max)
                evap_z = evap_z + 0.001
                call relative_depletion(th, th_sat, th_fc, th_wilt, th_dry, &
                     w_upper, w_lower, w_rel, rew, evap_z, dz, dz_sum, n_comp)
                w_check = f_wrel_exp * ((evap_z_max - evap_z) / (evap_z_max - evap_z_min))
             end do
          end if
          ! get stage 2 evaporation reduction coefficient
          kr = (exp(f_evap * w_rel) - 1) / (exp(f_evap) - 1)
          if (kr > 1) then
             kr = 1
          end if
          to_extract_stage_two = kr * edt
          call extract_water(to_extract, to_extract_stage_two, th, th_dry, dz_sum, evap_z, n_comp)
       end do
    end if
  end subroutine stage_two_evaporation
  
  subroutine soil_evaporation(prec, et_ref, irr, infl, &
       th, th_sat, th_fc, th_wilt, th_dry, &
       w_surf, w_stage_two, &
       evap_z, dz, dz_sum, &
       n_farm, n_crop, n_comp, n_cell, &
       timestep)
    
    integer, intent(in) :: timestep, n_farm, n_crop, n_comp, n_cell
    real(8), dimension(n_farm, n_crop, n_cell), intent(inout) :: w_stage_two, w_surf

    ! prepare soil evaporation stage two
    if (timestep == 1) then
       w_surf = 0
       evap_z = evap_z_min
       
       call get_evap_lyr_wc(th, th_sat, th_fc, th_wilt, th_dry, &
          w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry, &
          evap_z, dz, dz_sum, n_comp)

       w_stage_two = (w_evap_act - (w_evap_fc - rew)) / (w_evap_sat - (w_evap_fc - rew))
       ! NewCond.Wstage2 = round((100*NewCond.Wstage2))/100;       
       if (w_stage_two < 0) then
          w_stage_two = 0
       end if
    end if

    ! prepare soil evaporation stage one
    if (prec > 0 .or. (irr > 0 .and. .not. irr_method == 4)) then
       if (infl > 0) then
          w_surf = infl
          if (w_surf > rew) then
             w_surf = rew
          end if
          w_stage_two = 0
          evap_z = evap_z_min
       end if
    end if
    
    ! potential soil evaporation
    if (calendar_type == 1) then
       t_adj = dap - delayed_cds
    else if (calendar_type == 2) then
       t_adj = dap - delayed_gdds
    end if
    
    es_pot = potential_soil_evaporation_rate(et_ref, cc, cc_adj, &
         ccx_act, growing_season, premat_senes, kex, ccxw, fwcc, &
         t_adj, senescence)

    ! adjust potential soil evaporation for mulch
    es_pot_mulch = potential_soil_evaporation_rate_with_mulches( &
         mulches, growing_season, surface_storage, f_mulch, &
         mulch_pct_gs, mulch_pct_os, es_pot)

    ! adjust potential soil evaporation for irrigation
    es_pot_irr = potential_soil_evaporation_rate_with_irrigation( &
         prec, irr, irr_method, surface_storage, wet_surf)

    ! adjusted potential soil evaporation is the minimum value
    ! (effects of mulch and irrigation do not combine)
    es_pot = min(es_pot_mulch, es_pot_irr)

    call surface_evaporation(es_pot, es_act, surface_storage, rew, &
         w_stage_two, w_surf, evap_z)    

    to_extract = es_pot - es_act
    call stage_one_evaporation(to_extract, w_surf, es_act, w_surf, &
         evap_z, evap_z_min, th, dz, dz_sum, n_comp)

    call stage_two_evaporation(to_extract, evap_time_steps, th, &
         th_sat, th_fc, th_wilt, th_dry, evap_z, dz, dz_sum, rew, &
         evap_z, evap_z_max, evap_z_min, fwrel_exp)    
    
    e_pot = es_pot
  end subroutine soil_evaporation
  
end module soil_evaporation


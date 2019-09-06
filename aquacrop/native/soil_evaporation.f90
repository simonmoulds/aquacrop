module soil_evaporation
  use types
  implicit none

  real(real64), parameter :: zero = 0.0d0
  integer(int32), parameter :: one = 1
  integer(int32), parameter :: two = 2
  integer(int32), parameter :: four = 4

  ! TODO: compare against some best practice guidelines, e.g.
  ! https://www.fortran90.org/src/best-practices.html
  
contains


  
  ! Get array showing which compartments lie in the soil evaporative zone.
  !
  ! Input:
  !    evap_z     : depth of evaporative zone
  !    dz         : compartment thicknesses
  !    dz_sum     : accumulated compartment thicknesses
  !
  ! Output:
  !    comp_idx   : array showing which compartments lie in the evaporative zone
  !
  ! -------------------------------------------------------------------
  function get_comp_idx(evap_z, dz, dz_sum) result (comp_idx)
    
    real(real64), dimension(:), intent(in) :: dz, dz_sum
    real(real64), intent(in) :: evap_z
    
    real(real64) :: z_top
    integer(int32) :: i
    integer(int32) :: n_comp
    integer(int32), allocatable, dimension(:) :: comp_idx

    n_comp = size(dz, 1)
    allocate(comp_idx(n_comp))
    
    do i = 1, n_comp
       z_top = dz_sum(i) - dz(i)
       if (z_top < evap_z) then
          comp_idx(i) = 1
       else
          comp_idx(i) = 0
       end if       
    end do
    
  end function get_comp_idx


  
  ! Get the index of the deepest compartment in evaporation layer.
  !
  ! Input:
  !    evap_z : depth of soil evaporation layer (m)
  !    dz_sum : accumulated compartment depths (m)
  ! 
  ! Output:
  !    n_comp_max : index of deepest compartment
  ! 
  ! -------------------------------------------------------------------
  function get_max_comp_idx(evap_z, dz, dz_sum) result (max_comp_idx)

    real(real64), dimension(:), intent(in) :: dz, dz_sum
    real(real64), intent(in) :: evap_z
    integer(int32) :: i
    integer(int32) :: n_comp
    integer(int32) :: max_comp_idx

    n_comp = size(dz, 1)
    max_comp_idx = 1
    do i = 1, n_comp
       if (dz_sum(i) < evap_z) then
          max_comp_idx = max_comp_idx + 1
       end if       
    end do
    
  end function get_max_comp_idx



  ! Compute potential soil evaporation rate
  !
  ! Input:
  !    TODO
  !
  ! Output:
  !    es_pot : potential soil evaporation rate (mm d-1)
  ! 
  ! -------------------------------------------------------------------
  function pot_soil_evap(et_ref, cc, cc_adj, ccx_act, &
       growing_season, senescence, premat_senes, t_adj, &
       kex, ccxw, fwcc) result (es_pot)

    real(real64),   intent(in) :: et_ref, cc, cc_adj, ccx_act, kex, ccxw, fwcc
    integer(int32), intent(in) :: t_adj, growing_season, senescence, premat_senes
    real(real64) :: es_pot, es_pot_min, es_pot_max, mult, ccx_act_adj

    if ( growing_season == one ) then
       
       es_pot_max = kex * et_ref * (1 - ccxw * (fwcc / 100))
       es_pot = kex * (1 - cc_adj) * et_ref

       if ( t_adj > senescence .and. ccx_act > zero ) then
          if ( cc > (ccx_act / 2) ) then
             if ( cc > ccx_act ) then
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
          es_pot_min = max(es_pot_min, zero)
          es_pot = max(es_pot, es_pot_min)
          es_pot = min(es_pot, es_pot_max)

       else
          es_pot_max = es_pot_max
          es_pot = es_pot
       end if

       if ( premat_senes == one .and. es_pot > es_pot_max ) then
          es_pot = es_pot_max
       end if
       
    else
       es_pot = kex * et_ref       
    end if
    
  end function pot_soil_evap

  ! Compute water contents in soil evaporation layer.
  !
  ! Input:
  !    th, th_sat, th_fc, th_wilt th_dry : volumetric water contents (m3 m-3)
  !    evap_z : depth of soil evaporation layer (m)
  !    dz     : thickness of each soil compartment (m)
  !    dz_sum : accumulated compartment depth (m)
  !
  ! Output:
  !    w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp : depth of water in soil evaporation layer (mm)
  !
  ! -------------------------------------------------------------------
  subroutine get_evap_lyr_wc(th, th_sat, th_fc, th_wilt, th_dry, &
       w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry, &
       evap_z, dz, dz_sum)
    
    real(real64), dimension(:), intent(in) :: th, th_sat, th_fc, th_wilt, th_dry, dz, dz_sum
    real(real64), intent(in) :: evap_z
    real(real64), intent(out) :: w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry
    integer(int32) :: i, max_comp_idx
    real(real64) :: factor

    ! initialize water content to zero
    w_evap_act = 0
    w_evap_sat = 0
    w_evap_fc = 0
    w_evap_wp = 0
    w_evap_dry = 0

    ! get the index of the deepest compartment in the soil evaporation layer
    max_comp_idx = get_max_comp_idx(evap_z, dz, dz_sum)
    
    ! loop through compartments, calculating the depth of water in each
    do i = 1, max_comp_idx
       factor = 1 - (dz_sum(i) - evap_z) / dz(i)
       factor = min(factor, 1.0)
       factor = max(factor, 0.0)
       w_evap_act = w_evap_act + factor * 1000 * th(i) * dz(i)
       w_evap_sat = w_evap_sat + factor * 1000 * th_sat(i) * dz(i)
       w_evap_fc = w_evap_fc + factor * 1000 * th_fc(i) * dz(i)
       w_evap_wp = w_evap_wp + factor * 1000 * th_wilt(i) * dz(i)
       w_evap_dry = w_evap_dry + factor * 1000 * th_dry(i) * dz(i)
    end do
    
  end subroutine get_evap_lyr_wc

  ! Compute potential soil evaporation rate, adjusted for mulches
  !
  ! Input
  !    es_pot : potential soil evaporation rate
  !    ...
  !
  ! Output
  !    es_pot_mulch : potential soil evaporation rate, adjusted for mulches (mm d-1)
  !
  ! -------------------------------------------------------------------
  function pot_soil_evap_w_mul(es_pot, growing_season, surface_storage, &
       mulches, f_mulch, mulch_pct_gs, mulch_pct_os) result (es_pot_mul)

    integer(int32), intent(in) :: mulches, growing_season
    real(real64), intent(in) :: es_pot, surface_storage, f_mulch, mulch_pct_gs, mulch_pct_os
    real(real64) :: es_pot_mul

    if ( surface_storage < 0.000001 ) then
       
       if ( mulches == zero ) then
          es_pot_mul = es_pot
       else
          if ( growing_season == one ) then
             es_pot_mul = es_pot * (1 - f_mulch * (mulch_pct_gs / 100))
          else
             es_pot_mul = es_pot * (1 - f_mulch * (mulch_pct_os / 100))
          end if
       end if
       
    else
       es_pot_mul = es_pot
       
    end if
        
  end function pot_soil_evap_w_mul

  ! Compute potential soil evaporation rate, adjusted for irrigation
  !
  ! Input
  !    es_pot          : potential soil evaporation rate (mm d-1)
  !    prec            : precipitation rate (mm d-1)
  !    irr             : irrigation rate (mm d-1)
  !    irr_method      : irrigation method (1-4)
  !    surface_storage : depth of water in surface storage (i.e. because of bunds) (mm)
  !    wet_surf        : TODO
  !
  ! Output
  !    es_pot_irr : potential soil evaporation rate, adjusted for irrigation (mm d-1)
  !
  ! -------------------------------------------------------------------
  function pot_soil_evap_w_irr(es_pot, prec, irr, irr_method, &
       surface_storage, wet_surf) result (es_pot_irr)

    integer(int32), intent(in) :: irr_method
    real(real64), intent(in) :: es_pot, prec, irr, surface_storage, wet_surf
    real(real64) :: es_pot_irr

    if ( irr > zero .and. ( .not. irr_method == four ) ) then
       if ( prec > 1.0 .or. surface_storage > zero ) then
          es_pot_irr = es_pot
       else
          es_pot_irr = es_pot * (wet_surf / 100)
       end if
    else
       es_pot_irr = es_pot
    end if
        
  end function pot_soil_evap_w_irr


  
  ! Compute evaporation from surface storage
  !
  ! Input:
  !    es_pot          : potential soil evaporation (mm d-1)
  !    es_act          : actual soil evaporation
  !    rew             : readily evaporable(?) water (mm)
  !    surface_storage :
  !    w_surf          : 
  !    w_stage_two     : 
  !    evap_z          :
  !
  ! Output:
  !    es_act, surface_storage, w_surf, w_stage_two, evap_z : updated
  !    
  subroutine surf_evap(es_pot, es_act, surface_storage, rew, &
       w_surf, w_stage_two, evap_z, evap_z_min)

    real(real64), intent(in) :: es_pot, rew, evap_z_min
    real(real64), intent(inout) :: es_act, surface_storage, w_surf, w_stage_two, evap_z

    if ( surface_storage > zero ) then
       if ( surface_storage > es_pot ) then
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
    
  end subroutine surf_evap



  ! Extract water from the soil layer to meet soil evaporation demand.
  !
  ! Input:
  !    to_extract
  !    to_extract_stage
  !    es_act
  !    th
  !    th_dry
  !    dz, dz_sum
  !    evap_z
  !    evap_z_min
  !
  ! Output:
  !    to_extract, to_extract_stage, es_act, th
  !
  ! -------------------------------------------------------------------  
  subroutine extract_water(to_extract, to_extract_stage, es_act, th, th_dry, &
       dz, dz_sum, evap_z, evap_z_min)

    real(real64), dimension(:), intent(in) :: th_dry, dz, dz_sum
    real(real64), intent(in) :: evap_z, evap_z_min

    real(real64), dimension(:), intent(inout) :: th
    real(real64), intent(inout) :: to_extract, to_extract_stage, es_act
    
    real(real64) :: factor, w_dry, w, av_w
    integer(int32) :: comp, max_comp_idx
    
    ! get the index of the deepest compartment in the soil evaporation layer                                
    max_comp_idx = get_max_comp_idx(evap_z, dz, dz_sum)
    comp = 0
    do while ( to_extract_stage > 0 .and. comp < max_comp_idx )
       comp = comp + 1

       if ( dz_sum(comp) > evap_z_min ) then
          factor = 1 - (dz_sum(comp) - evap_z) / dz(comp)
          factor = min(factor, 1.0)
          factor = max(factor, 0.0)
          w_dry = 1000 * th_dry(comp) * dz(comp)
          w = 1000 * th(comp) * dz(comp)
          av_w = (w - w_dry) * factor
          av_w = max(av_w, 0.0)

          if (av_w > to_extract_stage) then
             es_act = es_act + to_extract_stage
             w = w - to_extract_stage                
             to_extract = to_extract - to_extract_stage
             to_extract_stage = 0
          else
             es_act = es_act + av_w
             to_extract_stage = to_extract_stage - av_w
             to_extract = to_extract - av_w
             w = w - av_w
          end if
          th(comp) = w / (1000 * dz(comp))
          
       end if
    end do
    
  end subroutine extract_water



  ! Compute soil evaporation.
  !
  ! Input:
  !    TODO
  ! 
  ! Output:
  !    TODO
  !
  ! -------------------------------------------------------------------
  subroutine soil_evap( &
       prec, &
       et_ref, es_act, e_pot, &
       irr, irr_method, &
       infl, &
       th, th_sat, th_fc, th_wilt, th_dry, &
       surface_storage, &
       wet_surf, w_surf, w_stage_two, &
       cc, cc_adj, ccx_act, &
       evap_z, evap_z_min, evap_z_max, &
       rew, kex, ccxw, fwcc, f_evap, f_wrel_exp, &
       dz, dz_sum, &
       mulches, f_mulch, mulch_pct_gs, mulch_pct_os, &
       growing_season, senescence, premat_senes, &
       calendar_type, dap, delayed_cds, delayed_gdds, &
       time_step, evap_time_steps)
    
    integer(int32), intent(in) :: calendar_type
    integer(int32), intent(in) :: time_step
    integer(int32), intent(in) :: evap_time_steps
    
    real(real64), dimension(:), intent(in) :: th_sat, th_fc, th_wilt, th_dry
    real(real64), dimension(:), intent(in) :: dz, dz_sum
    
    real(real64), intent(in) :: prec, et_ref
    real(real64), intent(in) :: infl
    real(real64), intent(in) :: irr, wet_surf
    real(real64), intent(in) :: rew
    real(real64), intent(in) :: evap_z_min, evap_z_max    
    real(real64), intent(in) :: cc, cc_adj, ccx_act
    real(real64), intent(in) :: kex, ccxw, fwcc, f_evap, f_wrel_exp
    real(real64), intent(in) :: f_mulch, mulch_pct_gs, mulch_pct_os
    
    integer(int32), intent(in) :: irr_method
    integer(int32), intent(in) :: dap, delayed_cds, delayed_gdds
    integer(int32), intent(in) :: mulches
    integer(int32), intent(in) :: growing_season
    integer(int32), intent(in) :: senescence, premat_senes
    
    real(real64), dimension(:), intent(inout) :: th
    real(real64), intent(inout) :: es_act, e_pot
    real(real64), intent(inout) :: surface_storage, w_surf, w_stage_two, evap_z
    
    real(real64) :: w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry
    real(real64) :: w_upper, w_lower, w_rel, w_check
    real(real64) :: es_pot, es_pot_mul, es_pot_irr
    real(real64) :: to_extract, to_extract_stage_one, to_extract_stage_two
    real(real64) :: edt
    real(real64) :: kr

    integer(int32) :: t_adj
    integer(int32) :: i
    
    ! prepare soil evaporation stage two
    if ( time_step == one ) then
       w_surf = 0
       evap_z = evap_z_min
       
       call get_evap_lyr_wc(th, th_sat, th_fc, th_wilt, th_dry, &
            w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry, &
            evap_z, dz, dz_sum)

       w_stage_two = (w_evap_act - (w_evap_fc - rew)) / (w_evap_sat - (w_evap_fc - rew))
       w_stage_two = max(w_stage_two, 0.0)
       
    end if

    ! prepare soil evaporation stage one if rainfall has occurred, or
    ! when irrigation is triggered and not in net irrigation mode.
    if ( prec > zero .or. ( irr > zero .and. ( .not. irr_method == four ) ) ) then
       if ( infl > zero ) then
          w_surf = infl
          w_surf = min(w_surf, rew)
          w_stage_two = 0
          evap_z = evap_z_min
       end if
    end if    
    
    ! calculate potential soil evaporation rate    
    if (calendar_type == one) then
       t_adj = dap - delayed_cds       
    else if (calendar_type == two) then
       t_adj = dap - delayed_gdds       
    end if
    
    es_pot = pot_soil_evap(et_ref, cc, cc_adj, ccx_act, &
         growing_season, senescence, premat_senes, t_adj, &
         kex, ccxw, fwcc)

    ! adjust potential soil evaporation for mulches
    es_pot_mul = pot_soil_evap_w_mul(es_pot, growing_season, &
         surface_storage, mulches, f_mulch, mulch_pct_gs, mulch_pct_os)

    ! adjust potential soil evaporation for irrigation
    es_pot_irr = pot_soil_evap_w_irr(es_pot, prec, irr, irr_method, &
       surface_storage, wet_surf)

    ! adjusted potential soil evaporation is the minimum value
    ! (effects of mulch and irrigation do not combine)
    es_pot = min(es_pot_mul, es_pot_irr)

    ! surface evaporation
    call surf_evap(es_pot, es_act, surface_storage, rew, &
       w_surf, w_stage_two, evap_z, evap_z_min)

    ! stage one evaporation
    to_extract = es_pot - es_act
    to_extract_stage_one = min(to_extract, w_surf)

    if ( to_extract_stage_one > zero ) then

       call extract_water(to_extract, to_extract_stage_one, es_act, th, th_dry, &
          dz, dz_sum, evap_z, evap_z_min)
       
       ! update surface evaporation layer water balance
       w_surf = w_surf - es_act
       if ( w_surf < zero .or. to_extract_stage_one > 0.0001 ) then
          w_surf = 0
       end if
       
       if ( w_surf < 0.0001 ) then
          
          call get_evap_lyr_wc(th, th_sat, th_fc, th_wilt, th_dry, &
               w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry, &
               evap_z, dz, dz_sum)

          w_stage_two = (w_evap_act - (w_evap_fc - rew)) / (w_evap_sat - (w_evap_fc - rew))
          w_stage_two = max(w_stage_two, 0.0)
          
       end if
    end if

    ! stage two evaporation
    if ( to_extract > zero ) then
       edt = to_extract / evap_time_steps

       do i = 1, evap_time_steps

          call get_evap_lyr_wc(th, th_sat, th_fc, th_wilt, th_dry, &
               w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry, &
               evap_z, dz, dz_sum)
          
          w_upper = w_stage_two * (w_evap_sat - (w_evap_fc - rew)) + (w_evap_fc - rew)
          w_lower = w_evap_dry
          w_rel = (w_evap_act - w_lower) / (w_upper - w_lower)
          
          if ( evap_z_max > evap_z_min ) then
             
             w_check = f_wrel_exp * ((evap_z_max - evap_z) / (evap_z_max - evap_z_min))
             
             do while ( w_rel < w_check .and. evap_z < evap_z_max )
                
                evap_z = evap_z + 0.001
                
                call get_evap_lyr_wc(th, th_sat, th_fc, th_wilt, th_dry, &
                     w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry, &
                     evap_z, dz, dz_sum)

                w_upper = w_stage_two * (w_evap_sat - (w_evap_fc - rew)) + (w_evap_fc - rew)
                w_lower = w_evap_dry
                w_rel = (w_evap_act - w_lower) / (w_upper - w_lower)
                w_check = f_wrel_exp * ((evap_z_max - evap_z) / (evap_z_max - evap_z_min))
                
             end do
          end if
          
          ! get stage 2 evaporation reduction coefficient
          kr = ( exp(f_evap * w_rel) - 1 ) / ( exp(f_evap) - 1 )
          kr = min(kr, 1.0)
          
          to_extract_stage_two = kr * edt
          
          call extract_water(to_extract, to_extract_stage_two, es_act, th, th_dry, &
             dz, dz_sum, evap_z, evap_z_min)
          
       end do
    end if
    
  end subroutine soil_evap
  
end module soil_evaporation


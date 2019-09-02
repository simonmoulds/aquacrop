module soil_evaporation
  implicit none

  ! https://www.fortran90.org/src/best-practices.html
  
contains


  
  ! Get array showing which compartments lie in the soil evaporative zone.
  !
  ! Input:
  !    evap_z     : depth of evaporative zone
  !    dz         : compartment thicknesses
  !    dz_sum     : accumulated compartment thicknesses
  !    n_farm, n_crop, n_comp, n_cell : array dimensions
  !
  ! Output:
  !    comp_idx   : array showing which compartments lie in the evaporative zone
  !
  ! -------------------------------------------------------------------
  function get_comp_idx(evap_z, dz, dz_sum, n_farm, n_crop, n_comp, n_cell) result (comp_idx)
    
    integer, intent(in) :: n_farm, n_crop, n_comp, n_cell
    real(8), dimension(n_comp), intent(in) :: dz, dz_sum
    real(8), dimension(n_farm,n_crop,n_cell), intent(in) :: evap_z
    real(8) :: z_top
    integer :: i
    integer, dimension(n_farm, n_crop, n_comp, n_cell) :: comp_idx
    
    comp_idx = 0
    do i = 1, n_comp
       z_top = dz_sum(i) - dz(i)
       where ( z_top < evap_z ) comp_idx(:,:,i,:) = 1
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
  function get_max_comp_idx(evap_z, dz, dz_sum, n_farm, n_crop, n_comp, n_cell) result (max_comp_idx)

    integer, intent(in) :: n_farm, n_crop, n_comp, n_cell
    real(8), dimension(n_comp), intent(in) :: dz, dz_sum
    real(8), dimension(n_farm,n_crop,n_cell), intent(in) :: evap_z
    integer :: i
    integer, dimension(n_farm, n_crop, n_cell) :: max_comp_idx

    max_comp_idx = 1
    do i = 1, n_comp
       where (dz_sum(i) < evap_z ) max_comp_idx = max_comp_idx + 1
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
  function pot_soil_evap_rate(et_ref, cc, cc_adj, ccx_act, &
       growing_season, senescence, premat_senes, t_adj, kex, ccxw, fwcc, &
       n_farm, n_crop, n_cell) result (es_pot)

    integer, intent(in) :: n_farm, n_crop, n_cell
    real(8), dimension(n_farm, n_crop, n_cell), intent(in) :: et_ref, cc, cc_adj, ccx_act, kex, ccxw, fwcc
    integer, dimension(n_farm, n_crop, n_cell), intent(in) :: t_adj, growing_season, senescence, premat_senes
    real(8), dimension(n_farm, n_crop, n_cell) :: es_pot, es_pot_min, es_pot_max, mult, ccx_act_adj

    es_pot = 0
    es_pot_min = 0
    es_pot_max = 0
    mult = 0
    ccx_act_adj = 0
    
    where ( growing_season == 1 )
       es_pot_max = kex * et_ref * (1 - ccxw * (fwcc / 100))
       es_pot = kex * (1 - cc_adj) * et_ref
       
       where (t_adj > senescence .and. ccx_act > 0)
          
          where (cc > (ccx_act / 2))

             where (cc > ccx_act)
                mult = 0
             elsewhere
                mult = (ccx_act - cc) / (ccx_act / 2)
             end where

          elsewhere
             mult = 1
          end where
          
          es_pot = es_pot * (1 - ccx_act * (fwcc / 100) * mult)
          ccx_act_adj = (1.72 * ccx_act) + (ccx_act ** 2) - 0.3 * (ccx_act ** 3)
          es_pot_min = kex * (1 - ccx_act_adj) * et_ref

          where (es_pot_min < 0)
             es_pot_min = 0
          end where
          
          where (es_pot < es_pot_min)
             es_pot = es_pot_min
          end where
          
          where (es_pot > es_pot_max)
             es_pot = es_pot_max
          end where          
          
       elsewhere
          es_pot_max = es_pot_max
          es_pot = es_pot
       end where

       where (premat_senes == 1 .and. es_pot > es_pot_max) es_pot = es_pot_max
       
    elsewhere
       es_pot = kex * et_ref
       
    end where
    
  end function pot_soil_evap_rate

  ! for testing:
  ! ------------
  ! function apply_max(arr, mx, nx, ny) result(arr_max)    
  !   integer, intent(in) :: nx, ny
  !   real(8), dimension(nx, ny), intent(in) :: arr
  !   real(8), intent(in) :: mx
  !   real(8), dimension(nx, ny) :: arr_max    
  !   arr_max = min(arr, mx)    
  ! end function apply_max
  
  
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
       evap_z, dz, dz_sum, n_farm, n_crop, n_comp, n_cell)
    
    integer, intent(in) :: n_farm, n_crop, n_comp, n_cell
    real(8), dimension(n_farm, n_crop, n_comp, n_cell), intent(in) :: th, th_sat, th_fc, th_wilt, th_dry
    real(8), dimension(n_farm, n_crop, n_cell), intent(in) :: evap_z
    real(8), dimension(n_comp), intent(in) :: dz, dz_sum
    integer :: i, glob_max_comp_idx
    real(8), dimension(n_farm, n_crop, n_cell), intent(out) :: w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry
    integer, dimension(n_farm, n_crop, n_cell) :: max_comp_idx
    real(8), dimension(n_farm, n_crop, n_cell) :: factor

    ! initialize water content to zero
    w_evap_act = 0
    w_evap_sat = 0
    w_evap_fc = 0
    w_evap_wp = 0
    w_evap_dry = 0

    ! get the index of the deepest compartment in the soil evaporation layer
    max_comp_idx = get_max_comp_idx(evap_z, dz, dz_sum, n_farm, n_crop, n_comp, n_cell)
    glob_max_comp_idx = maxval(max_comp_idx)
    
    ! loop through compartments, calculating the depth of water in each
    do i = 1, glob_max_comp_idx
       factor = 1 - (dz_sum(i) - evap_z) / dz(i)
       where (factor > 1) factor = 1
       where (factor < 0) factor = 0       
       w_evap_act = w_evap_act + factor * 1000 * th(:,:,i,:) * dz(i)
       w_evap_sat = w_evap_sat + factor * 1000 * th_sat(:,:,i,:) * dz(i)
       w_evap_fc = w_evap_fc + factor * 1000 * th_fc(:,:,i,:) * dz(i)
       w_evap_wp = w_evap_wp + factor * 1000 * th_wilt(:,:,i,:) * dz(i)
       w_evap_dry = w_evap_dry + factor * 1000 * th_dry(:,:,i,:) * dz(i)
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
       mulches, f_mulch, mulch_pct_gs, mulch_pct_os, n_farm, n_crop, &
       n_cell) result (es_pot_mul)

    integer, intent(in) :: n_farm, n_crop, n_cell
    integer, dimension(n_farm, n_crop, n_cell), intent(in) :: mulches, growing_season
    real(8), dimension(n_farm, n_crop, n_cell), intent(in) :: es_pot, surface_storage, f_mulch, mulch_pct_gs, mulch_pct_os
    real(8), dimension(n_farm, n_crop, n_cell) :: es_pot_mul

    where (surface_storage < 0.000001)
       where (mulches == 0)
          es_pot_mul = es_pot
       elsewhere
          where (growing_season == 1)
             es_pot_mul = es_pot * (1 - f_mulch * (mulch_pct_gs / 100))
          elsewhere
             es_pot_mul = es_pot * (1 - f_mulch * (mulch_pct_os / 100))
          end where
       end where
    elsewhere
       es_pot_mul = es_pot
    end where
        
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
       surface_storage, wet_surf, n_farm, n_crop, n_cell) result (es_pot_irr)

    integer, intent(in) :: n_farm, n_crop, n_cell
    integer, dimension(n_farm, n_crop, n_cell), intent(in) :: irr_method
    real(8), dimension(n_farm, n_crop, n_cell), intent(in) :: es_pot, prec, irr, surface_storage, wet_surf
    real(8), dimension(n_farm, n_crop, n_cell) :: es_pot_irr

    where (irr > 0 .and. .not. irr_method == 4)
       where (prec > 1 .or. surface_storage > 0)
          es_pot_irr = es_pot
       elsewhere
          es_pot_irr = es_pot * (wet_surf / 100)
       end where

    elsewhere
       es_pot_irr = es_pot
    end where
        
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
       w_surf, w_stage_two, evap_z, evap_z_min, n_farm, n_crop, n_cell)

    integer, intent(in) :: n_farm, n_crop, n_cell
    real(8), dimension(n_farm, n_crop, n_cell), intent(in) :: es_pot, rew, evap_z_min
    real(8), dimension(n_farm, n_crop, n_cell), intent(inout) :: es_act, surface_storage, w_surf, w_stage_two, evap_z

    where (surface_storage > 0)
       where (surface_storage > es_pot)
          es_act = es_pot          
          surface_storage = surface_storage - es_act
          
       elsewhere
          es_act = surface_storage
          surface_storage = 0
          w_surf = rew
          w_stage_two = 0
          evap_z = evap_z_min
       end where
       
    end where
    
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
  !    n_farm, n_crop, n_comp, n_cell
  !
  ! Output:
  !    to_extract, to_extract_stage, es_act, th
  !
  ! -------------------------------------------------------------------  
  subroutine extract_water(to_extract, to_extract_stage, es_act, th, th_dry, &
       dz, dz_sum, evap_z, evap_z_min, n_farm, n_crop, n_comp, n_cell)

    integer, intent(in) :: n_farm, n_crop, n_comp, n_cell
    real(8), dimension(n_farm, n_crop, n_comp, n_cell), intent(in) :: th_dry
    real(8), dimension(n_farm, n_crop, n_cell), intent(in) :: evap_z, evap_z_min
    real(8), dimension(n_comp), intent(in) :: dz, dz_sum

    real(8), dimension(n_farm, n_crop, n_cell), intent(inout) :: to_extract, to_extract_stage, es_act
    real(8), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: th
    
    real(8), dimension(n_farm, n_crop, n_cell) :: factor, w_dry, w, av_w
    integer, dimension(n_farm, n_crop, n_cell) :: max_comp_idx
    integer :: comp, glob_max_comp_idx
    
    ! get the index of the deepest compartment in the soil evaporation layer                                
    max_comp_idx = get_max_comp_idx(evap_z, dz, dz_sum, n_farm, n_crop, n_comp, n_cell)
    glob_max_comp_idx = maxval(max_comp_idx)
    comp = 0
    do while (any(to_extract_stage > 0) .and. comp < glob_max_comp_idx)
       comp = comp + 1

       where (dz_sum(comp) > evap_z_min)
          factor = 1 - (dz_sum(comp) - evap_z) / dz(comp)
          where (factor > 1) factor = 1
          where (factor < 0) factor = 0
          w_dry = 1000 * th_dry(:,:,comp,:) * dz(comp)
          w = 1000 * th(:,:,comp,:) * dz(comp)
          av_w = (w - w_dry) * factor
          where (av_w < 0) av_w = 0

          where (av_w > to_extract_stage)
             es_act = es_act + to_extract_stage
             w = w - to_extract_stage                
             to_extract = to_extract - to_extract_stage
             to_extract_stage = 0
          elsewhere
             es_act = es_act + av_w
             to_extract_stage = to_extract_stage - av_w
             to_extract = to_extract - av_w
             w = w - av_w
          end where             
          th(:,:,comp,:) = w / (1000 * dz(comp))
       end where
    end do       
    
  end subroutine extract_water
  
  ! subroutine stage_one_evaporation(to_extract, w_surf, es_act, w_surf, evap_z, evap_z_min, th, dz, dz_sum, n_comp)
  !   real(8), intent(in) :: es_act
  !   real(8), dimension(n_comp), intent(in) :: dz, dz_sum, th_dry
  !   real(8), intent(inout) :: to_extract, w_surf
  !   real(8), dimension(n_comp), intent(inout) :: th    
  !   real(8) :: extract_pot_stage_one, factor
  !   integer :: comp, n_comp_max

  !   ! extract water
  !   ! to_extract = es_pot - es_act
  !   to_extract_stage_one = min(to_extract, w_surf)
  !   call extract_water(to_extract, to_extract_stage_one, th, th_dry, dz_sum, evap_z, n_comp)
    
  !   ! update surface evaporation layer water balance
  !   if (extract_pot_stage_one > 0) then
  !      w_surf = w_surf - es_act
  !      if (w_surf < 0 .or. extract_pot_stage_one > 0.0001) then
  !         w_surf = 0
  !      end if
  !   end if    
  ! end subroutine stage_one_evaporation

  ! subroutine relative_depletion(th, th_sat, th_fc, th_wilt, th_dry, &
  !      w_upper, w_lower, w_rel, rew, evap_z, dz, dz_sum, n_comp)
    
  !   integer, intent(in) :: n_comp
  !   real(8), dimension(n_comp), intent(in) :: th, th_sat, th_fc, th_wilt, th_dry, dz, dz_sum
  !   real(8), intent(in) :: rew, evap_z
  !   real(8), intent(out) :: w_upper, w_lower, w_rel
  !   real(8) :: w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry
    
  !   call get_evap_lyr_wc(th, th_sat, th_fc, th_wilt, &
  !      th_dry, evap_z, dz, dz_sum, w_evap_act, w_evap_sat, &
  !      w_evap_fc, w_evap_wp, w_evap_dry, n_comp)
  !   w_upper = w_stage_two * (w_evap_sat - (w_evap_fc - rew)) + (w_evap_fc - rew)
  !   w_lower = w_evap_dry
  !   w_rel = (w_evap_act - w_lower) / (w_upper - w_lower)    
  ! end subroutine relative_depletion
    
  ! subroutine stage_two_evaporation(to_extract, evap_time_steps, th, th_sat, th_fc, th_wilt, th_dry, evap_z, dz, dz_sum, rew, evap_z, evap_z_max, evap_z_min, fwrel_exp)
  !   real(8), intent(inout) :: to_extract
  !   integer, intent(in) :: evap_time_steps
  !   real(8) :: w_upper, w_lower, w_rel

  !   if (to_extract > 0) then
  !      edt = to_extract / evap_time_steps
  !      do i, evap_time_steps
  !         call relative_depletion(th, th_sat, th_fc, th_wilt, th_dry, &
  !              w_upper, w_lower, w_rel, rew, evap_z, dz, dz_sum, n_comp)

  !         if (evap_z_max > evap_z_min) then
  !            w_check = f_wrel_exp * ((evap_z_max - evap_z) / (evap_z_max - evap_z_min))
  !            do while (w_rel < w_check .and. evap_z < evap_z_max)
  !               evap_z = evap_z + 0.001
  !               call relative_depletion(th, th_sat, th_fc, th_wilt, th_dry, &
  !                    w_upper, w_lower, w_rel, rew, evap_z, dz, dz_sum, n_comp)
  !               w_check = f_wrel_exp * ((evap_z_max - evap_z) / (evap_z_max - evap_z_min))
  !            end do
  !         end if
  !         ! get stage 2 evaporation reduction coefficient
  !         kr = (exp(f_evap * w_rel) - 1) / (exp(f_evap) - 1)
  !         if (kr > 1) then
  !            kr = 1
  !         end if
  !         to_extract_stage_two = kr * edt
  !         call extract_water(to_extract, to_extract_stage_two, th, th_dry, dz_sum, evap_z, n_comp)
  !      end do
  !   end if
  ! end subroutine stage_two_evaporation
  
  ! subroutine soil_evaporation(prec, et_ref, irr, infl, &
  !      th, th_sat, th_fc, th_wilt, th_dry, &
  !      w_surf, w_stage_two, &
  !      evap_z, dz, dz_sum, &
  !      n_farm, n_crop, n_comp, n_cell, &
  !      timestep)
    
  !   integer, intent(in) :: timestep, n_farm, n_crop, n_comp, n_cell
  !   real(8), dimension(n_farm, n_crop, n_cell), intent(inout) :: w_stage_two, w_surf

  !   ! prepare soil evaporation stage two
  !   if (timestep == 1) then
  !      w_surf = 0
  !      evap_z = evap_z_min
       
  !      call get_evap_lyr_wc(th, th_sat, th_fc, th_wilt, th_dry, &
  !         w_evap_act, w_evap_sat, w_evap_fc, w_evap_wp, w_evap_dry, &
  !         evap_z, dz, dz_sum, n_comp)

  !      w_stage_two = (w_evap_act - (w_evap_fc - rew)) / (w_evap_sat - (w_evap_fc - rew))
  !      ! NewCond.Wstage2 = round((100*NewCond.Wstage2))/100;       
  !      if (w_stage_two < 0) then
  !         w_stage_two = 0
  !      end if
  !   end if

  !   ! prepare soil evaporation stage one
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
    
  !   ! potential soil evaporation
  !   if (calendar_type == 1) then
  !      t_adj = dap - delayed_cds
  !   else if (calendar_type == 2) then
  !      t_adj = dap - delayed_gdds
  !   end if
    
  !   es_pot = potential_soil_evaporation_rate(et_ref, cc, cc_adj, &
  !        ccx_act, growing_season, premat_senes, kex, ccxw, fwcc, &
  !        t_adj, senescence)

  !   ! adjust potential soil evaporation for mulch
  !   es_pot_mulch = potential_soil_evaporation_rate_with_mulches( &
  !        mulches, growing_season, surface_storage, f_mulch, &
  !        mulch_pct_gs, mulch_pct_os, es_pot)

  !   ! adjust potential soil evaporation for irrigation
  !   es_pot_irr = potential_soil_evaporation_rate_with_irrigation( &
  !        prec, irr, irr_method, surface_storage, wet_surf)

  !   ! adjusted potential soil evaporation is the minimum value
  !   ! (effects of mulch and irrigation do not combine)
  !   es_pot = min(es_pot_mulch, es_pot_irr)

  !   call surface_evaporation(es_pot, es_act, surface_storage, rew, &
  !        w_stage_two, w_surf, evap_z)    

  !   to_extract = es_pot - es_act
  !   call stage_one_evaporation(to_extract, w_surf, es_act, w_surf, &
  !        evap_z, evap_z_min, th, dz, dz_sum, n_comp)

  !   call stage_two_evaporation(to_extract, evap_time_steps, th, &
  !        th_sat, th_fc, th_wilt, th_dry, evap_z, dz, dz_sum, rew, &
  !        evap_z, evap_z_max, evap_z_min, fwrel_exp)    
    
  !   e_pot = es_pot
  ! end subroutine soil_evaporation
  
end module soil_evaporation


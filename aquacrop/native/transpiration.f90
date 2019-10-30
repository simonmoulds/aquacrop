module transpiration
  use types
  use soil_evaporation, only: get_max_comp_idx
  use root_zone_water, only: update_root_zone_water
  use water_stress, only: relative_depletion  
  implicit none

contains

  subroutine update_root_fact( &
       root_fact, &
       root_depth, &
       max_comp_idx, &
       dz, &
       dz_sum &
       )

    real(real64), dimension(:), intent(inout) :: root_fact
    real(real64), intent(inout) :: root_depth
    integer(int32), intent(in) :: max_comp_idx
    real(real64), dimension(:), intent(in) :: dz
    real(real64), dimension(:), intent(in) :: dz_sum
    integer(int32) :: i
    root_fact = 0.
    do i = 1, max_comp_idx
       if ( dz_sum(i) > root_depth ) then
          root_fact(i) = 1. - ((dz_sum(i) - root_depth) / dz(i))
       else
          root_fact(i) = 1.
       end if
    end do
    root_fact = max(root_fact, 0.)
    root_fact = min(root_fact, 1.)
  end subroutine update_root_fact  
       
  subroutine update_max_sink_term( &
       sx_comp, &
       root_depth, &
       max_comp_idx, &
       r_cor, &
       sx_bot, &
       sx_top, &
       irr_method, &
       dz_sum &
       )

    real(real64), dimension(:), intent(inout) :: sx_comp
    real(real64), intent(in) :: root_depth
    integer(int32), intent(in) :: max_comp_idx
    ! real(real64), intent(in) :: z_root
    real(real64), intent(in) :: r_cor
    ! real(real64), intent(in) :: z_min
    real(real64), intent(in) :: sx_bot
    real(real64), intent(in) :: sx_top
    integer(int32), intent(in) :: irr_method
    ! real(real64), dimension(:), intent(in) :: dz
    real(real64), dimension(:), intent(in) :: dz_sum
    integer(int32) :: i
    ! integer(int32) :: n_comp
    ! integer(int32) :: max_comp_idx    
    ! real(real64) :: root_depth
    real(real64) :: sx_comp_bot
    real(real64) :: sx_comp_top
    ! real(real64), allocatable, dimension(:) :: root_fact
    
    ! n_comp = size(dz, 1)
    ! allocate(root_fact(n_comp))
    ! root_depth = max(z_root, z_min)
    ! root_depth = real(nint(root_depth * 100.)) / 100.       
    ! max_comp_idx = get_max_comp_idx(root_depth, dz, dz_sum)
    ! root_fact = 0.
    ! do i = 1, max_comp_idx
    !    if ( dz_sum(i) > root_depth ) then
    !       root_fact(i) = 1. - ((dz_sum(i) - root_depth) / dz(i))
    !    else
    !       root_fact(i) = 1.
    !    end if
    ! end do
    ! root_fact = max(root_fact, 0.)
    ! root_fact = min(root_fact, 1.)

    ! determine maximum sink term for each compartment
    sx_comp = 0.
    if ( irr_method == 4 ) then
       ! net irrigation mode
       do i = 1, max_comp_idx
          sx_comp(i) = (sx_top + sx_bot) / 2.
       end do
    else       
       ! maximum sink term declines linearly with depth
       sx_comp_bot = sx_top
       do i = 1, max_comp_idx
          sx_comp_top = sx_comp_bot
          if ( dz_sum(i) <= root_depth ) then
             sx_comp_bot = sx_bot * r_cor + ((sx_top - sx_bot * r_cor) * ((root_depth - dz_sum(i)) / root_depth))
          else
             sx_comp_bot = sx_bot * r_cor
          end if
          sx_comp(i) = (sx_comp_top + sx_comp_bot) / 2.
       end do
    end if
    
  end subroutine update_max_sink_term  
    
  subroutine update_aeration_stress( &
       ksa_aer, &
       aer_days, &
       thrz_act, &
       thrz_sat, &
       thrz_aer, &
       lag_aer &
       )

    real(real64), intent(inout) :: ksa_aer
    integer(int32), intent(inout) :: aer_days
    real(real64), intent(in) :: thrz_act
    real(real64), intent(in) :: thrz_sat
    real(real64), intent(in) :: thrz_aer
    integer(int32), intent(in) :: lag_aer
    real(real64) :: stress
    
    if ( thrz_act > thrz_aer ) then
       if ( aer_days < lag_aer ) then
          stress = 1. - ((thrz_sat - thrz_act) / (thrz_sat - thrz_aer))
          ksa_aer = 1. - ((aer_days / 3.) * stress)
       else if ( aer_days >= lag_aer ) then
          ksa_aer = (thrz_sat - thrz_act) / (thrz_sat - thrz_aer)
       end if
       aer_days = aer_days + 1
       aer_days = min(aer_days, lag_aer)
    else
       ksa_aer = 1.
       aer_days = 0
    end if
    
  end subroutine update_aeration_stress

  subroutine adjust_surf_trans_stress( &
       tr_pot, &
       aer_days, &
       ksw_stolin, &
       thrz_act, &
       thrz_sat, &
       thrz_aer, &
       lag_aer, &
       irr_method &
       )

    real(real64), intent(inout) :: tr_pot
    integer(int32), intent(inout) :: aer_days
    real(real64), intent(in) :: ksw_stolin
    real(real64), intent(in) :: thrz_act
    real(real64), intent(in) :: thrz_sat
    real(real64), intent(in) :: thrz_aer
    integer(int32), intent(in) :: lag_aer
    integer(int32), intent(in) :: irr_method
    real(real64) :: stress
    real(real64) :: ksa_aer
    real(real64) :: ks
    
    call update_aeration_stress( &
         ksa_aer, &
         aer_days, &
         thrz_act, &
         thrz_sat, &
         thrz_aer, &
         lag_aer &
         )       

    ! maximum stress effect
    ks = min(ksw_stolin, ksa_aer)

    ! update potential transpiration in root zone (unless in net irrigation mode)
    if ( irr_method /= 4 ) then
       tr_pot = tr_pot * ks
    end if
    
  end subroutine adjust_surf_trans_stress  
       
  subroutine update_surf_trans( &
       tr_pot0, &
       tr_pot_ns, &
       tr_pot, &
       tr_act0, &
       aer_days, &
       aer_days_comp, &
       age_days, &
       age_days_ns, &
       day_submrgd, &
       surface_storage, &
       cc, &
       et0, &
       max_canopy_cd, &
       kcb, &
       cc_adj, &
       cc_adj_ns, &
       ccxw, &
       ccxw_ns, &
       a_tr, &
       fage, &
       lag_aer, &
       co2_conc, &
       co2_refconc, &
       dap, &
       delayed_cds, &
       dz &
       )
    
    real(real64), intent(inout) :: tr_pot0
    real(real64), intent(inout) :: tr_pot_ns
    real(real64), intent(inout) :: tr_pot
    real(real64), intent(inout) :: tr_act0
    integer(int32), intent(inout) :: aer_days
    integer(int32), dimension(:), intent(inout) :: aer_days_comp
    integer(int32), intent(inout) :: age_days
    integer(int32), intent(inout) :: age_days_ns
    integer(int32), intent(inout) :: day_submrgd
    real(real64), intent(inout) :: surface_storage
    real(real64), intent(in) :: cc
    real(real64), intent(in) :: et0
    integer(int32), intent(in) :: max_canopy_cd
    real(real64), intent(in) :: kcb
    real(real64), intent(in) :: cc_adj
    real(real64), intent(in) :: cc_adj_ns
    real(real64), intent(in) :: ccxw
    real(real64), intent(in) :: ccxw_ns
    real(real64), intent(in) :: a_tr
    real(real64), intent(in) :: fage
    integer(int32), intent(in) :: lag_aer
    real(real64), intent(in) :: co2_conc
    real(real64), intent(in) :: co2_refconc    
    integer(int32), intent(in) :: dap
    integer(int32), intent(in) :: delayed_cds
    real(real64), dimension(:), intent(in) :: dz

    real(real64) :: f_sub
    real(real64) :: kcb_ns
    real(real64) :: kcb_adj
    integer(int32) :: n_comp
    integer(int32) :: i
    integer(int32) :: dap_adj
    n_comp = size(dz, 1)
    
    ! calculate potential transpiration
    if ( dap > max_canopy_cd ) then
       age_days_ns = dap - max_canopy_cd
    end if
    if ( age_days_ns > 5 ) then
       kcb_ns = kcb - ((real(age_days_ns) - 5.) * (fage / 100.)) * ccxw_ns
    else
       kcb_ns = kcb
    end if

    ! update crop coefficient for co2 concentration
    if ( co2_conc > co2_refconc ) then
       kcb_ns = kcb_ns * (1. - 0.05 * ((co2_conc - co2_refconc) / (550. - co2_refconc)))
    end if

    ! determine potential transpiration rate (no water stress)
    tr_pot_ns = kcb_ns * cc_adj_ns * et0

    ! potential prior water stress and/or delayed development
    dap_adj = dap - delayed_cds
    if ( dap_adj > max_canopy_cd ) then
       age_days = dap_adj - max_canopy_cd
    end if

    ! update crop coefficient for ageing of canopy
    if ( age_days > 5 ) then
       kcb_adj = kcb - ((age_days - 5) * (fage / 100.)) * ccxw
    else
       kcb_adj = kcb
    end if

    ! update crop coefficient for co2 concentration
    if ( co2_conc > co2_refconc ) then
       kcb_adj = kcb_adj * (1. - 0.05 * ((co2_conc - co2_refconc) / (550. - co2_refconc)))
    end if

    ! determine potential transpiration rate
    tr_pot0 = kcb_adj * (cc_adj) * et0
    if ( cc < ccxw ) then
       if ( ccxw > 0.001 .and. cc > 0.001 ) then
          tr_pot0 = tr_pot0 * ((cc / ccxw) ** a_tr)
       end if
    end if

    ! calculate surface layer transpiration
    if ( surface_storage > 0. .and. day_submrgd < lag_aer ) then
       day_submrgd = day_submrgd + 1
       do i = 1, n_comp
          aer_days_comp(i) = aer_days_comp(i) + 1
          aer_days_comp(i) = min(aer_days_comp(i), lag_aer)
       end do

       f_sub = 1. - (real(day_submrgd) / real(lag_aer))
       if ( surface_storage > (f_sub * tr_pot0) ) then
          ! transpiration occurs from surface storage
          surface_storage = surface_storage - (f_sub * tr_pot0)
          tr_act0 = f_sub * tr_pot0             
       else
          tr_act0 = 0.
       end if

       if ( tr_act0 < (f_sub * tr_pot0) ) then
          ! more water can be extracted from soil profile for transpiration
          tr_pot = (f_sub * tr_pot0) - tr_act0
       else
          tr_pot = 0
       end if

    else
       tr_pot = tr_pot0
       tr_act0 = 0.
    end if
    
  end subroutine update_surf_trans
  
  subroutine update_transpiration( &
       tr_pot0, &               ! potential surface transpiration (mm)
       tr_pot_ns, &             ! potential transpiration, no stresses (mm)
       tr_act, &                ! actual transpiration (mm)
       tr_act0, &               ! actual surface transpiration (mm)
       t_pot, &                 ! potential transpiration (mm)
       tr_ratio, &              ! ratio between actual and potential transpiration (-)
       aer_days, &
       aer_days_comp, &
       th, &                    ! soil moisture (m3/m3)
       thrz_act, &
       thrz_sat, &
       thrz_fc, &
       thrz_wilt, &
       thrz_dry, &
       thrz_aer, &
       taw, &
       dr, &
       age_days, &
       age_days_ns, &
       day_submrgd, &
       surface_storage, &
       irr_net, &               ! net irrigation (mm)
       irr_net_cum, &           ! cumulative net irrigation (mm)
       cc, &                    ! canopy cover
       et0, &                   ! reference evapotranspiration (mm)
       th_sat, &                ! soil moisture at saturation (m3/m3)
       th_fc, &                 ! soil moisture at field capacity (m3/m3)
       th_wilt, &               ! soil moisture at wilting point (m3/m3)
       th_dry, &                ! soil moisture when soil is dry (m3/m3)
       max_canopy_cd, &         ! days after planting at which maximum canopy is achieved
       kcb, &
       ksw_stolin, &
       cc_adj, &
       cc_adj_ns, &
       cc_prev, &
       ccxw, &
       ccxw_ns, &
       z_root, &                ! rooting depth (mm)
       r_cor, &                 ! ????
       z_min, &                 ! minimum rooting depth (mm)
       a_tr, &
       aer, &
       fage, &
       lag_aer, &
       sx_bot, &
       sx_top, &
       et_adj, &
       p_lo2, &
       p_up2, &
       f_shape_w2, &
       irr_method, &            ! method use for irrigation
       net_irr_smt, &           ! soil moisture target if net irrigation method is used (mm)
       co2_conc, &              ! Carbon dioxide concentration
       co2_refconc, &           ! Reference carbon dioxide concentration
       dap, &                   ! days after planting
       delayed_cds, &           ! 
       dz, &                    ! soil compartment depth (m)
       dz_sum, &                ! cumulative soil compartment depth (m)
       layer_ix, &              ! index mapping layers to compartments
       growing_season &         ! logical indicating whether the growing season is active
       )
    
    real(real64), intent(inout) :: tr_pot0
    real(real64), intent(inout) :: tr_pot_ns
    real(real64), intent(inout) :: tr_act
    real(real64), intent(inout) :: tr_act0
    real(real64), intent(inout) :: t_pot
    real(real64), intent(inout) :: tr_ratio    
    integer(int32), intent(inout) :: aer_days
    integer(int32), dimension(:), intent(inout) :: aer_days_comp
    real(real64), dimension(:), intent(inout) :: th 
    real(real64), intent(inout) :: thrz_act
    real(real64), intent(inout) :: thrz_sat
    real(real64), intent(inout) :: thrz_fc
    real(real64), intent(inout) :: thrz_wilt
    real(real64), intent(inout) :: thrz_dry
    real(real64), intent(inout) :: thrz_aer
    real(real64), intent(inout) :: taw
    real(real64), intent(inout) :: dr
    integer(int32), intent(inout) :: age_days
    integer(int32), intent(inout) :: age_days_ns
    integer(int32), intent(inout) :: day_submrgd
    real(real64), intent(inout) :: surface_storage
    real(real64), intent(inout) :: irr_net
    real(real64), intent(inout) :: irr_net_cum
    real(real64), intent(inout) :: cc
        
    real(real64), intent(in) :: et0
    real(real64), dimension(:), intent(in) :: th_sat
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: th_wilt
    real(real64), dimension(:), intent(in) :: th_dry
    integer(int32), intent(in) :: max_canopy_cd
    real(real64), intent(in) :: kcb
    real(real64), intent(in) :: ksw_stolin

    real(real64), intent(in) :: cc_adj
    real(real64), intent(in) :: cc_adj_ns
    real(real64), intent(in) :: cc_prev
    real(real64), intent(in) :: ccxw
    real(real64), intent(in) :: ccxw_ns

    real(real64), intent(in) :: z_root
    real(real64), intent(in) :: r_cor
    
    real(real64), intent(in) :: z_min
    real(real64), intent(in) :: a_tr
    real(real64), intent(in) :: aer
    real(real64), intent(in) :: fage
    integer(int32), intent(in) :: lag_aer
    real(real64), intent(in) :: sx_bot
    real(real64), intent(in) :: sx_top
    integer(int32), intent(in) :: et_adj
    real(real64), intent(in) :: p_lo2
    real(real64), intent(in) :: p_up2
    real(real64), intent(in) :: f_shape_w2

    integer(int32), intent(in) :: irr_method
    real(real64), intent(in) :: net_irr_smt
    real(real64), intent(in) :: co2_conc
    real(real64), intent(in) :: co2_refconc
    
    integer(int32), intent(in) :: dap
    integer(int32), intent(in) :: delayed_cds
    real(real64), dimension(:), intent(in) :: dz
    real(real64), dimension(:), intent(in) :: dz_sum
    integer(int32), dimension(:), intent(in) :: layer_ix
    integer(int32), intent(in) :: growing_season

    ! real(real64) :: d_rel2
    real(real64) :: root_depth
    real(real64) :: w_rel
    real(real64) :: tr_pot
    real(real64) :: th_to_extract
    real(real64) :: th_taw
    real(real64) :: th_crit
    real(real64) :: sink
    ! real(real64) :: sx_comp_bot
    ! real(real64) :: sx_comp_top
    real(real64) :: p_up_sto
    ! real(real64) :: ksw_stolin
    ! real(real64) :: ksa_aer
    real(real64) :: ks
    real(real64) :: f_sub
    real(real64) :: f_aer
    real(real64) :: dwc
    real(real64) :: aer_comp
    real(real64) :: kcb_ns
    real(real64) :: kcb_adj
    real(real64) :: to_extract
    real(real64) :: ks_comp
    real(real64) :: p_rel
    integer(int32) :: i
    integer(int32) :: lyri
    integer(int32) :: pre_lyri
    integer(int32) :: max_comp_idx
    integer(int32) :: n_comp
    integer(int32) :: dap_adj
    integer(int32) :: comp
    real(real64), allocatable, dimension(:) :: root_fact
    real(real64), allocatable, dimension(:) :: sx_comp
    
    n_comp = size(dz, 1)
    allocate(root_fact(n_comp))
    allocate(sx_comp(n_comp))
    
    if ( growing_season == 1 ) then
       
       call update_surf_trans( &
            tr_pot0, &
            tr_pot_ns, &
            tr_pot, &
            tr_act0, &
            aer_days, &
            aer_days_comp, &
            age_days, &
            age_days_ns, &
            day_submrgd, &
            surface_storage, &
            cc, &
            et0, &
            max_canopy_cd, &
            kcb, &
            cc_adj, &
            cc_adj_ns, &
            ccxw, &
            ccxw_ns, &
            a_tr, &
            fage, &
            lag_aer, &
            co2_conc, &
            co2_refconc, &
            dap, &
            delayed_cds, &
            dz &
            )
       
       ! call update_root_zone_water( &
       !      thrz_act, &
       !      thrz_sat, &
       !      thrz_fc, &
       !      thrz_wilt, &
       !      thrz_dry, &
       !      thrz_aer, &
       !      taw, &
       !      dr, &
       !      th, &
       !      th_sat, &
       !      th_fc, &
       !      th_wilt, &
       !      th_dry, &
       !      aer, &
       !      z_root, &
       !      z_min, &
       !      dz, &
       !      dz_sum, &
       !      layer_ix &
       !      )

       ! d_rel2 = relative_depletion( &
       !      dr, &
       !      p_lo2, &
       !      p_up2, &
       !      taw &
       !      )
       ! ksw_stolin = 1. - d_rel2
       call adjust_surf_trans_stress( &
            tr_pot, &
            aer_days, &
            ksw_stolin, &
            thrz_act, &
            thrz_sat, &
            thrz_aer, &
            lag_aer, &
            irr_method &
            )
       
       root_depth = max(z_root, z_min)
       root_depth = real(nint(root_depth * 100.)) / 100.       
       max_comp_idx = get_max_comp_idx(root_depth, dz, dz_sum)

       call update_max_sink_term( &
            sx_comp, &
            root_depth, &
            max_comp_idx, &
            r_cor, &
            sx_bot, &
            sx_top, &
            irr_method, &
            dz_sum &
            )

       call update_root_fact( &
            root_fact, &
            root_depth, &
            max_comp_idx, &
            dz, &
            dz_sum &
            )
       
       ! extract water
       to_extract = tr_pot
       comp = 0
       tr_act = 0
       do while ( to_extract > 0. .and. comp < max_comp_idx )
          comp = comp + 1
          lyri = layer_ix(comp)

          ! get total available water for compartment
          th_taw = th_fc(lyri) - th_wilt(lyri)
          if ( et_adj == 1 ) then
             p_up_sto = p_up2 + (0.04 * (5. - et0)) * log10(10. - 9. * p_up2)
          end if
          
          th_crit = th_fc(lyri) - (th_taw * p_up_sto)
          if ( th(comp) > th_crit ) then
             ks_comp = 1.
          else if ( th(comp) > th_wilt(lyri) ) then
             ! transpiration is affected by water stress
             w_rel = (th_fc(lyri) - th(comp)) / (th_fc(lyri) - th_wilt(lyri))
             p_rel = (w_rel - p_up2) / (p_lo2 - p_up2)
             if ( p_rel <= 0. ) then
                ks_comp = 1.
             else if ( p_rel >= 1 ) then
                ks_comp = 0.
             else
                ks_comp = 1. - ((exp(p_rel * f_shape_w2) - 1) / (exp(f_shape_w2) - 1.))
             end if
             ks_comp = min(ks_comp, 1.)
             ks_comp = max(ks_comp, 0.)
          else
             ! no transpiration is possible from compartment as water
             ! content does not exceed wilting point
             ks_comp = 0.
          end if

          ! adjust compartment stress factor for aeration stress
          if ( day_submrgd >= lag_aer ) then
             ! full aeration stress- no transpiration possible from compartment
             aer_comp = 0.
          else if ( th(comp) > (th_sat(lyri) - (aer / 100)) ) then
             ! increment aeration stress days counter
             aer_days_comp(comp) = aer_days_comp(comp) + 1
             if ( aer_days_comp(comp) >= lag_aer ) then
                aer_days_comp(comp) = lag_aer
                f_aer = 0.
             else
                f_aer = 1.
             end if
             
             ! calculate aeration stress factor
             aer_comp = (th_sat(lyri) - th(comp)) / (th_sat(lyri) - (th_sat(lyri) - (aer / 100.)))
             aer_comp = max(aer_comp, 0.)
             aer_comp = (f_aer + (aer_days_comp(comp) - 1) * aer_comp) / (f_aer + aer_days_comp(comp) - 1)
          else
             ! no aeration stress as number of submerged days does not
             ! exceed threshold for initiation of aeration stress
             aer_comp = 1.
             aer_days_comp(comp) = 0.
          end if
          
          th_to_extract = (to_extract / 1000.) / dz(comp)
          if ( irr_method == 4 ) then
             ! don't reduce compartment sink for stomatal water stress if in net irrigation mode. Stress only occurs due to deficient aeration conditions
             sink = aer_comp * sx_comp(comp) * root_fact(comp)
          else
             if ( ks_comp == aer_comp ) then
                sink = ks_comp * sx_comp(comp) * root_fact(comp)
             else
                sink = min(ks_comp, aer_comp) * sx_comp(comp) * root_fact(comp)
             end if
          end if

          sink = min(sink, th_to_extract)
             
          ! limit extraction to avoid compartment water content dropping
          ! below air dry
          if ( (th(comp) - sink) < th_dry(lyri) ) then
             sink = th(comp) - th_dry(lyri)
          end if
          sink = max(sink, 0.)

          ! update water content in compartment
          th(comp) = th(comp) - sink
          ! update amount of water to extract
          to_extract = to_extract - (sink * 1000. * dz(comp))
          ! update actual transpiration
          tr_act = tr_act + (sink * 1000. * dz(comp))
       end do
       
       ! add net irrigation water requirement if this mode is specified
       if ( irr_method == 4 .and. tr_pot > 0. ) then
          irr_net = 0.
          call update_root_zone_water( &
               thrz_act, &
               thrz_sat, &
               thrz_fc, &
               thrz_wilt, &
               thrz_dry, &
               thrz_aer, &
               taw, &
               dr, &
               th, &
               th_sat, &
               th_fc, &
               th_wilt, &
               th_dry, &
               aer, &
               z_root, &
               z_min, &
               dz, &
               dz_sum, &
               layer_ix &
               )
          
          th_crit = thrz_wilt + ((net_irr_smt / 100.) * (thrz_fc - thrz_wilt))
          ! check if root zone water content is below net irrigation trigger
          if ( thrz_act < th_crit ) then
             pre_lyri = 0
             do i = 1, max_comp_idx
                lyri = layer_ix(i)
                if ( lyri > pre_lyri ) then
                   th_crit = th_wilt(lyri) + ((net_irr_smt / 100.) * (th_fc(lyri) - th_wilt(lyri)))
                   pre_lyri = lyri
                end if
                ! determine necessary change in water content in
                ! compartments to reach critical water content
                dwc = root_fact(i) * (th_crit - th(i)) * 1000. * dz(i)
                th(i) = th(i) + (dwc / (1000. * dz(i)))
                ! update net irrigation counter
                irr_net = irr_net + dwc
             end do
          end if
          ! update net irrigation counter for the growing season
          irr_net_cum = irr_net_cum + irr_net
       else if ( irr_method == 4 .and. tr_pot <= 0 ) then
          ! no net irrigation as potential transpiration is zero
          irr_net = 0.
       else
          ! no net irrigation as not in net irrigation mode
          irr_net = 0.
          irr_net_cum = 0.
       end if

       ! add any surface transpiration to root zone total       
       tr_act = tr_act + tr_act0

       ! feedback with canopy cover development
       if ( (cc - cc_prev) > 0.005 .and. tr_act == 0. ) then
          cc = cc_prev
       end if

       ! update transpiration ratio
       if ( tr_pot0 > 0. ) then
          if ( tr_act < tr_pot0 ) then
             tr_ratio = tr_act / tr_pot0
          else
             tr_ratio = 1.
          end if
       else
          tr_ratio = 1.
       end if
       tr_ratio = min(tr_ratio, 1.)
       tr_ratio = max(tr_ratio, 0.)
    else
       tr_act = 0.
       tr_pot0 = 0.
       tr_pot_ns = 0.
       irr_net = 0.
       irr_net_cum = 0.
    end if
    ! store potential transpiration for irrigation calculations on next day
    t_pot = tr_pot0
    
  end subroutine update_transpiration
end module transpiration

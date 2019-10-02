module harvest_index
  use types
  implicit none

  real(real64), parameter :: pi = 3.14159265359
  
contains

  subroutine update_harvest_index( &
       hi_ref, &
       pct_lag_phase, &
       yield_form, &
       cc_prev, &
       cc_min, &
       ccx, &
       hi_ini, &
       hi0, &
       higc, &
       hi_start, &
       hi_start_cd, &
       t_lin_switch, &
       dhi_linear, &
       gdd_cum, &
       dap, &
       delayed_cds, &
       delayed_gdds, &
       crop_type, &
       calendar_type, &
       growing_season &
       )
    
    real(real64), intent(inout) :: hi_ref
    real(real64), intent(inout) :: pct_lag_phase
    integer(int32), intent(inout) :: yield_form
    real(real64), intent(in) :: cc_prev
    real(real64), intent(in) :: cc_min
    real(real64), intent(in) :: ccx
    real(real64), intent(in) :: hi_ini
    real(real64), intent(in) :: hi0
    real(real64), intent(in) :: higc
    real(real64), intent(in) :: hi_start
    integer(int32), intent(in) :: hi_start_cd
    real(real64), intent(in) :: t_lin_switch
    real(real64), intent(in) :: dhi_linear
    real(real64), intent(in) :: gdd_cum
    integer(int32), intent(in) :: dap
    integer(int32), intent(in) :: delayed_cds
    real(real64), intent(in) :: delayed_gdds
    integer(int32), intent(in) :: crop_type
    integer(int32), intent(in) :: calendar_type
    integer(int32), intent(in) :: growing_season

    integer(int32) :: hi_t
    real(real64) :: t_adj
    
    if ( growing_season == 1 ) then
       
       ! check if in yield formation period
       if ( calendar_type == 1 ) then
          t_adj = dap - delayed_cds
       else if ( calendar_type == 2 ) then
          t_adj = gdd_cum - delayed_gdds
       end if

       if ( t_adj > hi_start ) then
          yield_form = 1
       else
          yield_form = 0
       end if

       hi_t = dap - delayed_cds - hi_start_cd - 1
       if ( hi_t <= 0 ) then
          hi_ref = 0.
          pct_lag_phase = 0.
       else
          if ( cc_prev <= (cc_min * ccx) ) then
             hi_ref = hi_ref
          else             
             if ( crop_type == 1 .or. crop_type == 2 ) then
                ! if crop type is a leafy vegetable or root/tuber, then
                ! proceed with logistic growth (no linear switch)
                pct_lag_phase = 100.
                hi_ref = (hi_ini * hi0) / (hi_ini + (hi0 - hi_ini) * exp(-higc * hi_t))
                if ( hi_ref >= (0.9799 * hi0) ) then
                   hi_ref = hi0
                end if
             else if ( crop_type == 3 ) then
                ! if crop type is fruit/grain producing, check for linear switch
                if ( hi_t < t_lin_switch ) then
                   ! not yet reached linear switch point, so proceed with logistic build-up
                   pct_lag_phase = 100. * (hi_t / t_lin_switch)
                   hi_ref = (hi_ini * hi0) / (hi_ini + (hi0 - hi_ini) * exp(-higc * hi_t))
                else
                   ! linear switch point has been reached
                   pct_lag_phase = 100.
                   ! logistic portion
                   hi_ref = (hi_ini * hi0) / (hi_ini + (hi0 - hi_ini) * exp(-higc * t_lin_switch))
                   ! linear portion
                   hi_ref = hi_ref + (dhi_linear * (hi_t - t_lin_switch))
                end if
             end if

             ! limit hi_ref and round off computed value
             if ( hi_ref > hi0 ) then
                hi_ref = hi0
             else if ( hi_ref <= (hi_ini + 0.004) ) then
                hi_ref = 0.
             else if ( (hi0 - hi_ref) < 0.004 ) then
                hi_ref = hi0
             end if
          end if
       end if
    else
       ! reference harvest index is zero outside growing season
       hi_ref = 0.
    end if
    
  end subroutine update_harvest_index

  subroutine hi_adj_pre_anthesis( &
       f_pre, &
       b, &
       b_ns, &
       dhi_pre, &
       cc &
       )

    real(real64), intent(inout) :: f_pre
    real(real64), intent(in) :: b
    real(real64), intent(in) :: b_ns
    real(real64), intent(in) :: dhi_pre
    real(real64), intent(in) :: cc
    real(real64) :: br
    real(real64) :: br_range
    real(real64) :: br_upp
    real(real64) :: br_low
    real(real64) :: br_top
    real(real64) :: ratio_low
    real(real64) :: ratio_upp

    ! parameters
    br = b / b_ns
    br_range = log(dhi_pre) / 5.62
    br_upp = 1.
    br_low = 1. - br_range
    br_top = br_upp - (br_range / 3.)

    ! biomass ratios
    ratio_low = (br - br_low) / (br_top - br_low)
    ratio_upp = (br - br_top) / (br_upp - br_top)

    ! calculate adjustment factor
    if ( br >= br_low .and. br < br_top ) then
       f_pre = 1. + (((1 + sin((1.5 - ratio_low) * pi)) / 2.) * (dhi_pre / 100.))
    else if ( br > br_top .and. br <= br_upp ) then
       f_pre = 1. + (((1 + sin((0.5 + ratio_upp) * pi)) / 2.) * (dhi_pre / 100.))
    else
       f_pre = 1.
    end if
    
    if ( cc <= 0.01 ) then
       ! no green canopy cover left at start of flowering so no
       ! harvestable crop will develop
       f_pre = 0.
    end if
    
  end subroutine hi_adj_pre_anthesis

  subroutine hi_adj_pol( &
       f_pol, &
       hi_t, &
       flowering_cd, &
       cc, &
       cc_min, &
       ksw_pol, &
       kst_polc, &
       kst_polh, &
       exc &
       )

    real(real64), intent(inout) :: f_pol
    integer(int32), intent(in) :: hi_t
    integer(int32), intent(in) :: flowering_cd
    real(real64), intent(in) :: cc
    real(real64), intent(in) :: cc_min
    real(real64), intent(in) :: ksw_pol
    real(real64), intent(in) :: kst_polc
    real(real64), intent(in) :: kst_polh
    real(real64), intent(in) :: exc 
    real(real64) :: frac_flow
    integer(int32) :: t1
    integer(int32) :: t2
    real(real64) :: t1pct
    real(real64) :: t2pct
    real(real64) :: f
    real(real64) :: f1
    real(real64) :: f2
    real(real64) :: ks
    real(real64) :: df_pol
    
    if ( hi_t == 0 ) then
       ! no flowering
       frac_flow = 0.
    else if ( hi_t > 0 ) then
       ! fractional flowering on previous day
       t1 = hi_t - 1
       if ( t1 == 0 ) then
          f1 = 0
       else
          t1pct = 100. * (real(t1) / real(flowering_cd))
          t1pct = min(t1pct, 100.)
          f1 = 0.00558 * exp(0.63 * log(t1pct)) - (0.000969 * t1pct) - 0.00383
       end if
       f1 = max(f1, 0.)
       ! fractional flowering on current day
       t2 = hi_t
       if ( t2 == 0 ) then
          f2 = 0
       else
          t2pct = 100. * (real(t2) / real(flowering_cd))
          t2pct = min(t2pct, 100.)
          f2 = 0.00558 * exp(0.63 * log(t2pct)) - (0.000969 * t2pct) - 0.00383
       end if
       f2 = max(f2, 0.)
       ! weight values
       if ( abs(f1 - f2) < 0.0000001 ) then
          f = 0.
       else
          f = 100. * ((f1 + f2) / 2.) / flowering_cd
       end if
       frac_flow = f
    end if
    ! calculate pollination adjustment for current day
    if ( cc < cc_min ) then
       ! no pollination can occur as cc is smaller than minimum threshold
       df_pol = 0.
    else
       ks = min(ksw_pol, kst_polc, kst_polh)
       df_pol = ks * frac_flow * (1 + (exc / 100.))
    end if
    ! calculate pollination adjustment to date
    f_pol = f_pol + df_pol
    f_pol = min(f_pol, 1.)
    
  end subroutine hi_adj_pol

  subroutine hi_adj_post_anthesis( &
       f_post, &
       fpost_dwn, &
       fpost_upp, &
       s_cor1, &
       s_cor2, &
       f_pre, &
       cc, &
       ksw_exp, &
       ksw_sto, &
       canopy_dev_end_cd, &
       hi_start_cd, &
       hi_end_cd, &
       yld_form_cd, &
       a_hi, &
       b_hi, &
       dap, &
       delayed_cds &
       )

    real(real64), intent(inout) :: f_post
    real(real64), intent(inout) :: fpost_dwn
    real(real64), intent(inout) :: fpost_upp
    real(real64), intent(inout) :: s_cor1
    real(real64), intent(inout) :: s_cor2
    real(real64), intent(in) :: f_pre
    real(real64), intent(in) :: cc
    real(real64), intent(in) :: ksw_exp
    real(real64), intent(in) :: ksw_sto
    integer(int32), intent(in) :: canopy_dev_end_cd
    integer(int32), intent(in) :: hi_start_cd
    integer(int32), intent(in) :: hi_end_cd
    integer(int32), intent(in) :: yld_form_cd
    real(real64), intent(in) :: a_hi
    real(real64), intent(in) :: b_hi
    integer(int32), intent(in) :: dap
    integer(int32), intent(in) :: delayed_cds

    integer(int32) :: t_adj
    real(real64) :: tmax1, tmax2
    real(real64) :: d_cor
    real(real64) :: day_cor
    
    ! adjust dap
    t_adj = dap - delayed_cds

    ! adjustment for leaf expansion
    tmax1 = canopy_dev_end_cd - hi_start_cd
    if ( t_adj <= (canopy_dev_end_cd + 1) &
         .and. tmax1 > 0 &
         .and. f_pre > 0.99 &
         .and. cc > 0.001 &
         .and. a_hi > 0. ) then
       d_cor = ( 1. + (1. - ksw_exp) / a_hi )
       s_cor1 = s_cor1 + (d_cor / tmax1)
       day_cor = t_adj - 1 - hi_start_cd
       fpost_upp = (tmax1 / day_cor) * s_cor1
    end if

    ! adjustment for stomatal closure
    tmax2 = yld_form_cd
    if ( t_adj <= (hi_end_cd + 1) &
         .and. tmax2 > 0 &
         .and. f_pre > 0.99 &
         .and. cc > 0.001 &
         .and. b_hi > 0.) then

       d_cor = exp(0.1 * log(ksw_sto)) * (1. - (1. - ksw_sto) / b_hi)
       s_cor2 = s_cor2 + (d_cor / tmax2)
       day_cor = t_adj - 1 - hi_start_cd       
       fpost_dwn = (tmax2 / day_cor) * s_cor2
    end if
    
    ! determine total multiplier
    if ( tmax1 == 0 .and. tmax2 == 0 ) then
       f_post = 1.
    else
       if ( tmax2 == 0 ) then
          f_post = fpost_upp
       else
          if ( tmax1 == 0 ) then
             f_post = fpost_dwn
          else if ( tmax1 <= tmax2 ) then
             f_post = fpost_dwn * (((tmax1 * fpost_upp) + (tmax2 - tmax1)) / tmax2)
          else
             f_post = fpost_upp * (((tmax2 * fpost_dwn) + (tmax1 - tmax2)) / tmax1)
          end if
       end if
    end if

  end subroutine hi_adj_post_anthesis

  subroutine adjust_harvest_index( &
       hi_adj, &
       pre_adj, &
       f_pre, &
       f_pol, &
       f_post, &
       fpost_dwn, &
       fpost_upp, &
       s_cor1, &
       s_cor2, &
       yield_form, &
       hi_ref, &
       hi0, &
       dhi0, &
       b, &
       b_ns, &
       dhi_pre, &
       cc, &
       cc_min, &
       ksw_exp, &
       ksw_sto, &
       ksw_pol, &
       kst_polc, &
       kst_polh, &
       canopy_dev_end_cd, &
       hi_start_cd, &
       hi_end_cd, &
       yld_form_cd, &
       flowering_cd, &
       a_hi, &
       b_hi, &
       exc, &
       dap, &
       delayed_cds, &
       crop_type, &
       growing_season &
       )

    real(real64), intent(inout) :: hi_adj
    integer(int32), intent(inout) :: pre_adj
    real(real64), intent(inout) :: f_pre
    real(real64), intent(inout) :: f_pol
    real(real64), intent(inout) :: f_post
    real(real64), intent(inout) :: fpost_dwn
    real(real64), intent(inout) :: fpost_upp
    real(real64), intent(inout) :: s_cor1
    real(real64), intent(inout) :: s_cor2
    integer(int32), intent(in) :: yield_form
    real(real64), intent(in) :: hi_ref
    real(real64), intent(in) :: hi0
    real(real64), intent(in) :: dhi0
    real(real64), intent(in) :: b
    real(real64), intent(in) :: b_ns
    real(real64), intent(in) :: dhi_pre
    real(real64), intent(in) :: cc
    real(real64), intent(in) :: cc_min
    real(real64), intent(in) :: ksw_exp
    real(real64), intent(in) :: ksw_sto
    real(real64), intent(in) :: ksw_pol
    real(real64), intent(in) :: kst_polc
    real(real64), intent(in) :: kst_polh
    integer(int32), intent(in) :: canopy_dev_end_cd
    integer(int32), intent(in) :: hi_start_cd
    integer(int32), intent(in) :: hi_end_cd
    integer(int32), intent(in) :: yld_form_cd
    integer(int32), intent(in) :: flowering_cd
    real(real64), intent(in) :: a_hi
    real(real64), intent(in) :: b_hi
    real(real64), intent(in) :: exc
    integer(int32), intent(in) :: dap
    integer(int32), intent(in) :: delayed_cds
    integer(int32), intent(in) :: crop_type
    integer(int32), intent(in) :: growing_season

    integer(int32) :: hi_t
    real(real64) :: hi_i
    real(real64) :: hi_max
    real(real64) :: hi_mult
    
    if ( growing_season == 1 ) then
       hi_i = hi_ref
       hi_t = dap - delayed_cds - hi_start_cd - 1
       if ( yield_form == 1 .and. hi_t >= 0 ) then
          ! root/tuber or fruit/grain crops
          if ( crop_type == 2 .or. crop_type == 3 ) then
             ! determine adjustment for water stress before anthesis
             if ( pre_adj == 0 ) then
                pre_adj = 1
                call hi_adj_pre_anthesis( &
                     f_pre, &
                     b, &
                     b_ns, &
                     dhi_pre, &
                     cc &
                     )
             end if
             
             ! determine adjustment for crop pollination failure
             if ( crop_type == 3) then
                if ( hi_t > 0 .and. hi_t <= flowering_cd ) then
                   call hi_adj_pol( &
                        f_pol, &
                        hi_t, &
                        flowering_cd, &
                        cc, &
                        cc_min, &
                        ksw_pol, &
                        kst_polc, &
                        kst_polh, &
                        exc &
                        )
                   
                end if
                hi_max = f_pol * hi0
             else
                hi_max = hi0
             end if

             ! determine adjustments for post-anthesis water stress
             if ( hi_t > 0 ) then
                call hi_adj_post_anthesis( &
                     f_post, &
                     fpost_dwn, &
                     fpost_upp, &
                     s_cor1, &
                     s_cor2, &
                     f_pre, &
                     cc, &
                     ksw_exp, &
                     ksw_sto, &
                     canopy_dev_end_cd, &
                     hi_start_cd, &
                     hi_end_cd, &
                     yld_form_cd, &
                     a_hi, &
                     b_hi, &
                     dap, &
                     delayed_cds &
                     )
             end if

             ! limit hi to maximum allowable increase due to pre/post
             ! anthesis water stress combinations
             hi_mult = f_pre * f_post
             hi_mult = min(hi_mult, 1. + (dhi0 / 100.))
             
             ! determine hi on current day, adjusted for stress effects
             if ( hi_max > hi_i ) then
                hi_adj = hi_mult * hi_i
             else
                hi_adj = hi_mult * hi_max
             end if
             
          else if ( crop_type == 1 ) then
             hi_adj = hi_i
          end if
       else
          ! hi_i = hi
          hi_adj = hi_adj
       end if
       ! hi = hi_i
       hi_adj = hi_adj
       
    else
       ! hi = 0
       hi_adj = 0
    end if
    
  end subroutine adjust_harvest_index
  
end module harvest_index

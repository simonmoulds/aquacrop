module pre_irr
  use types
  use soil_evaporation, only: get_max_comp_idx
  implicit none

contains

  subroutine update_pre_irr( &
       pre_irr, &
       th, &
       irr_method, &
       dap, &
       z_root, &
       z_min, &
       net_irr_smt, &
       th_fc, &
       th_wilt, &
       dz, &
       dz_sum, &
       layer_ix &
       )

    real(real64), intent(inout) :: pre_irr
    real(real64), dimension(:), intent(inout) :: th
    integer(int32), intent(in) :: irr_method
    integer(int32), intent(in) :: dap
    real(real64), intent(in) :: z_root
    real(real64), intent(in) :: z_min
    real(real64), intent(in) :: net_irr_smt
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: th_wilt
    real(real64), dimension(:), intent(in) :: dz
    real(real64), dimension(:), intent(in) :: dz_sum
    real(real64), dimension(:), intent(in) :: layer_ix
    
    integer(int32) :: max_comp_idx
    integer(int32) :: i
    integer(int32) :: lyri
    real(real64) :: root_depth
    real(real64) :: th_crit

    ! TODO: update irr_net_cum here, not in irrigation
    
    if ( irr_method /= 4 .or. dap /= 0 ) then
       pre_irr = 0.
    else
       root_depth = max(z_root, z_min)
       max_comp_idx = get_max_comp_idx(root_depth, dz, dz_sum)
       pre_irr = 0.
       do i = 1, max_comp_idx
          lyri = layer_ix(i)
          th_crit = th_wilt(lyri) + ((net_irr_smt / 100.) * (th_fc(lyri) - th_wilt(lyri)))
          if ( th(i) < th_crit ) then
             pre_irr = pre_irr + ((th_crit - th(i)) * 1000. * dz(i))
             th(i) = th_crit
          end if
       end do
    end if
    
  end subroutine update_pre_irr
  
end module pre_irr

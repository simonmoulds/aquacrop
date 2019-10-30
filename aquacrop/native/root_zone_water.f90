module root_zone_water
  use types
  use soil_evaporation, only: get_max_comp_idx
  implicit none

contains
  subroutine update_root_zone_water( &
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

    real(real64), intent(inout) :: thrz_act
    real(real64), intent(inout) :: thrz_sat
    real(real64), intent(inout) :: thrz_fc
    real(real64), intent(inout) :: thrz_wilt
    real(real64), intent(inout) :: thrz_dry
    real(real64), intent(inout) :: thrz_aer
    real(real64), intent(inout) :: taw
    real(real64), intent(inout) :: dr
    real(real64), dimension(:), intent(in) :: th
    real(real64), dimension(:), intent(in) :: th_sat
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: th_wilt
    real(real64), dimension(:), intent(in) :: th_dry
    real(real64), intent(in) :: aer
    real(real64), intent(in) :: z_root
    real(real64), intent(in) :: z_min
    real(real64), dimension(:), intent(in) :: dz
    real(real64), dimension(:), intent(in) :: dz_sum
    integer(int32), dimension(:), intent(in) :: layer_ix
    
    integer(int32) :: max_comp_idx
    integer(int32) :: i
    integer(int32) :: lyri
    real(real64) :: wr
    real(real64) :: wr_sat
    real(real64) :: wr_fc
    real(real64) :: wr_wilt
    real(real64) :: wr_dry
    real(real64) :: wr_aer
    real(real64) :: root_depth
    real(real64) :: factor
    
    root_depth = max(z_root, z_min)
    root_depth = real(nint(root_depth * 1000.)) / 1000.
    max_comp_idx = get_max_comp_idx(root_depth, dz, dz_sum)
    wr = 0.
    wr_sat = 0.
    wr_fc = 0.
    wr_wilt = 0.
    wr_dry = 0.
    wr_aer = 0.

    do i = 1, max_comp_idx
       lyri = layer_ix(i)
       if ( dz_sum(i) > root_depth ) then
          factor = 1. - ((dz_sum(i) - root_depth) / dz(i))
       else
          factor = 1.
       end if

       ! actual water storage in root zone (mm)
       wr = wr + (factor * 1000. * th(i) * dz(i))
       wr_sat = wr_sat + (factor * 1000 * th_sat(lyri) * dz(i))
       wr_fc = wr_fc + (factor * 1000 * th_fc(lyri) * dz(i))
       wr_wilt = wr_wilt + (factor * 1000 * th_wilt(lyri) * dz(i))
       wr_dry = wr_dry + (factor * 1000 * th_dry(lyri) * dz(i))
       wr_aer = wr_aer + (factor * 1000 * aer * dz(i))
    end do

    wr = max(wr, 0.)

    ! actual root zone water content (m3/m3)
    thrz_act = wr / (root_depth * 1000)
    thrz_sat = wr_sat / (root_depth * 1000)
    thrz_fc = wr_fc / (root_depth * 1000)
    thrz_wilt = wr_wilt / (root_depth * 1000)
    thrz_dry = wr_dry / (root_depth * 1000)
    thrz_aer = wr_aer / (root_depth * 1000)

    ! calculate total available water (mm)
    taw = wr_fc - wr_wilt
    taw = max(taw, 0.)

    ! calculate root zone depletion (mm)
    dr = wr_fc - wr
    dr = max(dr, 0.)

  end subroutine update_root_zone_water
end module root_zone_water


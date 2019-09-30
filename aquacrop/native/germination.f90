module germination
  use types
  use soil_evaporation, only: get_max_comp_idx
  implicit none

contains
  
  subroutine update_germ( &
       germ, &
       delayed_cds, &
       delayed_gdds, &
       th, &
       th_fc, &
       th_wilt, &
       z_germ, &
       germ_thr, &
       dz, &
       dz_sum, &
       layer_ix, &
       growing_season &
       )

    integer(int32), intent(inout) :: germ
    integer(int32), intent(inout) :: delayed_cds
    real(real64), intent(inout) :: delayed_gdds
    real(real64), dimension(:), intent(in) :: th
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: th_wilt
    real(real64), intent(in) :: z_germ
    real(real64), intent(in) :: germ_thr
    real(real64), dimension(:), intent(in) :: dz
    real(real64), dimension(:), intent(in) :: dz_sum
    integer(int32), dimension(:), intent(in) :: layer_ix
    integer(int32), intent(in) :: growing_season

    integer(int32) :: n_comp
    integer(int32) :: i
    integer(int32) :: lyri
    integer(int32) :: max_comp_idx
    real(real64) :: wr
    real(real64) :: wr_fc
    real(real64) :: wr_wp
    real(real64) :: wc_prop
    real(real64) :: factor
    
    if ( growing_season == 1 ) then
       
       ! max_comp_idx = get_max_comp_idx(z_germ, dz, dz_sum)
       n_comp = size(dz, 1)
       max_comp_idx = 1
       do i = 1, n_comp
          if (dz_sum(i) < z_germ) then
             max_comp_idx = max_comp_idx + 1
          end if       
       end do

       ! calculate water content in top soil layer
       wr = 0.
       wr_fc = 0.
       wr_wp = 0.
       
       ! NOTE: a similar algorithm is soil_evaporation.f90
       do i = 1, max_comp_idx
          lyri = layer_ix(i)
          factor = 1 - (dz_sum(i) - z_germ) / dz(i)
          factor = min(factor, 1.0)
          factor = max(factor, 0.0)
          wr = wr + factor * 1000 * th(i) * dz(i)
          wr_fc = wr_fc + factor * 1000 * th_fc(lyri) * dz(i)
          wr_wp = wr_wp + factor * 1000 * th_wilt(lyri) * dz(i)
       end do

       ! calculate proportional water content
       wc_prop = 1. - ((wr_fc - wr) / (wr_fc - wr_wp))

       ! check if water content is above germination threshold
       if ( wc_prop >= germ_thr .and. germ == 0 ) then
          germ = 1
       end if

       ! increment delayed growth time counters if
       ! germination is yet to occur
       if ( germ == 0 ) then
          delayed_cds = delayed_cds + 1
          delayed_gdds = delayed_gdds + 1
       end if

    else
       germ = 0
       delayed_cds = 0
       delayed_gdds = 0
    end if
    
  end subroutine update_germ

end module germination



  

module rainfall_partition
  use types
  use soil_evaporation, only: get_max_comp_idx
  implicit none
  
contains
  subroutine update_rain_part( &
       runoff, &
       infl, &
       prec, &
       th, &
       days_submrgd, &
       bund, &
       z_bund, &
       th_fc, &
       th_wilt, &
       cn, &
       adj_cn, &
       z_cn, &
       cn_bot, &
       cn_top, &
       dz, &
       dz_sum, &
       layer_ix &
       )

    real(real64), intent(inout) :: runoff 
    real(real64), intent(inout) :: infl
    real(real64), intent(in) :: prec
    real(real64), dimension(:), intent(in) :: th
    integer(int32), intent(inout) :: days_submrgd
    integer(int32), intent(in) :: bund
    real(real64), intent(in) :: z_bund
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: th_wilt
    integer(int32), intent(in) :: cn
    integer(int32), intent(in) :: adj_cn
    real(real64), intent(in) :: z_cn
    real(real64), intent(in) :: cn_bot
    real(real64), intent(in) :: cn_top
    real(real64), dimension(:), intent(in) :: dz
    real(real64), dimension(:), intent(in) :: dz_sum
    integer(int32), dimension(:), intent(in) :: layer_ix

    integer(int32) :: max_comp_idx
    integer(int32) :: i
    integer(int32) :: lyri
    real(real64) :: z_bot
    real(real64) :: xx
    real(real64) :: wx
    real(real64), allocatable, dimension(:) :: w_rel
    real(real64) :: thh
    real(real64) :: wet_top
    real(real64) :: s
    real(real64) :: term
    integer(int32) :: cn_adj
    
    if ( bund == 0 .or. z_bund < 0.001 ) then
       days_submrgd = 0
       if ( adj_cn == 1 ) then
          ! adjust curve number for antecedent moisture
          max_comp_idx = get_max_comp_idx(z_cn, dz, dz_sum)
          allocate(w_rel(max_comp_idx))
          xx = 0.
          do i = 1, max_comp_idx
             z_bot = min(dz_sum(i), z_cn)
             wx = 1.016 * (1. - exp(-4.16 * (z_bot / z_cn)))
             w_rel(i) = wx - xx
             w_rel(i) = max(w_rel(i), 0.)
             w_rel(i) = min(w_rel(i), 1.)
             xx = wx
          end do
          
          ! calculate relative wetness of top soil
          wet_top = 0.
          do i = 1, max_comp_idx
             lyri = layer_ix(i)
             thh = max(th_wilt(lyri), th(i))
             wet_top = wet_top + (w_rel(i) * ((thh - th_wilt(lyri)) / (th_fc(lyri) - th_wilt(lyri))))
          end do
          wet_top = max(wet_top, 0.)
          wet_top = min(wet_top, 1.)
          cn_adj = nint(cn_bot + (cn_top - cn_bot) * wet_top)
       else
          cn_adj = cn
       end if

       ! partition rainfall into runoff and infiltration
       s = (25400. / real(cn_adj)) - 254.
       term = prec - ((5. / 100.) * s)
       if ( term <= 0. ) then
          runoff = 0.
          infl = prec
       else
          runoff = (term ** 2) / (prec + (1. - (5. / 100.)) * s)
          infl = prec - runoff
       end if
    else
       ! bunds on field, therefore no surface runoff
       runoff = 0.
       infl = prec
    end if
    
    ! clean up
    deallocate(w_rel)
    
  end subroutine update_rain_part
  
end module rainfall_partition

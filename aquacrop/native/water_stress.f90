module water_stress
  use types
  implicit none
  
contains

  function relative_depletion( &
       dr, &
       p_lo, &
       p_up, &
       taw &
       ) result(d_rel)

    real(real64), intent(in) :: dr
    real(real64), intent(in) :: p_lo
    real(real64), intent(in) :: p_up
    real(real64), intent(in) :: taw
    real(real64) :: d_rel    
    if ( dr <= (p_up * taw) ) then
       d_rel = 0.
    else if ( dr > (p_up * taw) .and. dr < (p_lo * taw) ) then
       d_rel = 1. - ((p_lo - (dr / taw)) / (p_lo - p_up))
    else if ( dr >= (p_lo * taw) ) then
       d_rel = 1.
    end if
    
  end function relative_depletion
         
  subroutine update_water_stress( &
       ksw_exp, &
       ksw_sto, &
       ksw_sen, &
       ksw_pol, &
       ksw_stolin, &
       dr, &
       taw, &
       et_ref, &
       et_adj, &
       t_early_sen, &
       p_up1, &
       p_up2, &
       p_up3, &
       p_up4, &
       p_lo1, &
       p_lo2, &
       p_lo3, &
       p_lo4, &
       f_shape_w1, &
       f_shape_w2, &
       f_shape_w3, &
       f_shape_w4, &
       beta &
       )

    real(real64), intent(inout) :: ksw_exp
    real(real64), intent(inout) :: ksw_sto
    real(real64), intent(inout) :: ksw_sen
    real(real64), intent(inout) :: ksw_pol
    real(real64), intent(inout) :: ksw_stolin
    real(real64), intent(in) :: dr
    real(real64), intent(in) :: taw
    real(real64), intent(in) :: et_ref
    integer(int32), intent(in) :: et_adj
    real(real64), intent(in) :: t_early_sen
    real(real64), intent(in) :: p_up1
    real(real64), intent(in) :: p_up2
    real(real64), intent(in) :: p_up3
    real(real64), intent(in) :: p_up4
    real(real64), intent(in) :: p_lo1
    real(real64), intent(in) :: p_lo2
    real(real64), intent(in) :: p_lo3
    real(real64), intent(in) :: p_lo4
    real(real64), intent(in) :: f_shape_w1
    real(real64), intent(in) :: f_shape_w2
    real(real64), intent(in) :: f_shape_w3
    real(real64), intent(in) :: f_shape_w4
    integer(int32), intent(in) :: beta

    real(real64), dimension(4) :: p_up
    real(real64), dimension(4) :: p_lo
    real(real64), dimension(4) :: f_shape_w
    real(real64), dimension(4) :: d_rel
    real(real64), dimension(3) :: ks
    integer(int32) :: i

    p_up = (/p_up1, p_up2, p_up3, p_up4 /)
    p_lo = (/p_lo1, p_lo2, p_lo3, p_lo4 /)
    f_shape_w = (/f_shape_w1, f_shape_w2, f_shape_w3, f_shape_w4/)
    
    ! adjust stress thresholds for et_ref on current day
    if ( et_adj == 1 ) then
       do i = 1, 3
          p_up(i) = p_up(i) + (0.04 * (5 - et_ref)) * log10(10 - 9 * p_up(i))
          p_lo(i) = p_lo(i) + (0.04 * (5 - et_ref)) * log10(10 - 9 * p_lo(i))
       end do
    end if

    ! adjust senescence threshold if senescence is triggered
    if ( beta == 1 .and. t_early_sen > 0 ) then
       p_up(3) = p_up(3) * (1 - beta / 100)
    end if

    ! limit so that 0 <= p_up|lo <= 1
    p_up = max(p_up, 0.)
    p_up = min(p_up, 1.)
    p_lo = max(p_lo, 0.)
    p_lo = min(p_lo, 1.)

    ! calculate relative depletion
    do i = 1, 4
       d_rel(i) = relative_depletion(dr, p_lo(i), p_up(i), taw)
    end do

    ! calculate root zone water stress coefficients
    do i = 1, 3
       ks(i) = 1 - ((exp(d_rel(i) * f_shape_w(i)) - 1) / (exp(f_shape_w(i)) - 1))
    end do

    ! water stress coefficient for leaf expansion
    ksw_exp = ks(1)
    ! water stress coefficient for stomatal closure
    ksw_sto = ks(2)
    ! water stress coefficient for senescence
    ksw_sen = ks(3)
    ! water stress coefficient for pollination failure
    ksw_pol = 1 - d_rel(4)
    ! mean water stress coefficient for stomatal closure
    ksw_stolin = 1 - d_rel(2)
       
  end subroutine update_water_stress
  
end module water_stress

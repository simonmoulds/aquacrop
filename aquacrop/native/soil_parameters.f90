module soil_parameters
  use types
  implicit none

  real(real64), parameter :: zero = 0.0d0
  
contains

  subroutine compute_curve_number_limits( &
       cn_bot, &
       cn_top, &
       cn &
       )

    real(real64), intent(inout) :: cn_bot
    real(real64), intent(inout) :: cn_top
    real(real64), intent(in) :: cn

    cn_bot = real(nint(1.4 * exp(-14. * log(10.)) + (0.507 * cn) - (0.00374 * cn ** 2) + (0.0000867 * cn ** 3)))
    cn_top = real(nint(5.6 * exp(-14. * log(10.)) + (2.33 * cn) - (0.0209 * cn ** 2) + (0.000076 * cn ** 3)))
    
  end subroutine compute_curve_number_limits

  subroutine compute_rew( &
       rew, &
       th_fc, &
       th_dry, &
       evap_z_surf &
       )
    
    real(real64), intent(inout) :: rew
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: th_dry
    real(real64), intent(in) :: evap_z_surf

    rew = real(nint(1000. * (th_fc(1) - th_dry(1)) * evap_z_surf))

  end subroutine compute_rew

  ! subroutine compute_wrel( &
  !      w_rel, &
  !      z_cn, &
  !      dz_sum &
  !      )

  !   real(real64), dimension(:), intent(inout) :: w_rel
  !   real(real64), intent(in) :: z_cn
  !   real(real64), dimension(:), intent(in) :: dz_sum
  !   integer(int32) :: n, i
  !   real(real64) :: dz_sum_i, xx, wx
    
  !   n = size(dz_sum)

  !   xx = 0.
  !   do i = 1, n

  !      if ( dz_sum(i) > z_cn ) then
  !         dz_sum_i = z_cn
  !      else
  !         dz_sum_i = dz_sum(i)
  !      end if
  !      wx = 1.016 * (1 - exp(-4.16 * (dz_sum_i / z_cn)))
  !      w_rel(i) = wx - xx
  !      if ( w_rel(i) < 0 ) then
  !         w_rel(i) = 0.
  !      else if ( w_rel(i) > 1 ) then
  !         w_rel(i) = 1.
  !      end if
  !      xx = wx
       
  !   end do

  ! end subroutine compute_wrel

  subroutine adjust_zgerm( &
       z_germ, &
       dz_sum &
       )

    real(real64), intent(inout) :: z_germ
    real(real64), dimension(:), intent(in) :: dz_sum
    integer(int32) :: n

    n = size(dz_sum)
    
    if ( z_germ > dz_sum(n) ) then
       z_germ = dz_sum(n)
    end if
    
  end subroutine adjust_zgerm

  subroutine compute_soil_parameters( &
       cn_bot, &
       cn_top, &
       rew, &
       z_germ, &
       cn, &
       th_fc, &
       th_dry, &
       evap_z_surf, &
       z_cn, &
       dz_sum &
       )
    
    real(real64), intent(inout) :: cn_bot
    real(real64), intent(inout) :: cn_top
    real(real64), intent(inout) :: rew
    real(real64), intent(inout) :: z_germ    
    real(real64), intent(in) :: cn
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: th_dry
    real(real64), intent(in) :: evap_z_surf
    real(real64), intent(in) :: z_cn
    real(real64), dimension(:), intent(in) :: dz_sum

    call compute_curve_number_limits( &
         cn_bot, &
         cn_top, &
         cn &
         )
    
    call compute_rew( &
         rew, &
         th_fc, &
         th_dry, &
         evap_z_surf &
         )

    call adjust_zgerm( &
         z_germ, &
         dz_sum &
         )

  end subroutine compute_soil_parameters  
    
end module soil_parameters


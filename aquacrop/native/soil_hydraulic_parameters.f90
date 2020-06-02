module soil_hydraulic_parameters
  use types
  implicit none

  real(real64), parameter :: zero = 0.0d0
  
contains

  subroutine compute_cap_rise_parameters( &
       a_cr, &
       b_cr, &
       th_fc, &
       th_sat, &
       th_wilt, &
       k_sat, &
       water_table &
       )

    real(real64), dimension(:), intent(inout) :: a_cr
    real(real64), dimension(:), intent(inout) :: b_cr
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: th_sat
    real(real64), dimension(:), intent(in) :: th_wilt
    real(real64), dimension(:), intent(in) :: k_sat
    integer(int32), intent(in) :: water_table
    integer(int32) :: n
    integer(int32) :: i
    
    n = size(th_fc)
    
    if ( water_table == 1 ) then

       do i = 1, n
          
          if ( th_wilt(i) >= 0.04 .and. th_wilt(i) <= 0.15 .and. th_fc(i) >= 0.09 &
               .and. th_fc(i) <= 0.28 .and. th_sat(i) >= 0.32 .and. th_sat(i) <= 0.51 ) then
             
             if ( k_sat(i) >= 200. .and. k_sat(i) <= 2000. ) then
                a_cr(i) = -0.3112 - (k_sat(i) * 1E-05)
                b_cr(i) = -1.4936 + (0.2416 * log(k_sat(i)))
                
             else if ( k_sat(i) < 200. ) then
                a_cr(i) = -0.3112 - (200. * 1E-05)
                b_cr(i) = -1.4936 + (0.2416 * log(200.))
                
             else if ( k_sat(i) > 2000. ) then
                a_cr(i) = -0.3112 - (2000. * 1E-05)
                b_cr(i) = -1.4936 + (0.2416 * log(2000.))
             end if

          else if ( th_wilt(i) >= 0.06 .and. th_wilt(i) <= 0.20 .and. th_fc(i) >= 0.23 &
               .and. th_fc(i) <= 0.42 .and. th_sat(i) >= 0.42 .and. th_sat(i) <= 0.55 ) then

             if ( k_sat(i) >= 100. .and. k_sat(i) <= 750. ) then                
                a_cr(i) = -0.4986 + (k_sat(i) * 9. * 1E-05)
                b_cr(i) = -2.132 + (0.4778 * log(k_sat(i)))

             else if ( k_sat(i) < 100. ) then
                a_cr(i) = -0.4986 + (100. * 9. * 1E-05)
                b_cr(i) = -2.132 + (0.4778 * log(100.))

             else if ( k_sat(i) > 750. ) then
                a_cr(i) = -0.4986 + (750. * 9. * 1E-05)
                b_cr(i) = -2.132 + (0.4778 * log(750.))

             end if

          else if ( th_wilt(i) >= 0.16 .and. th_wilt(i) <= 0.34 .and. th_fc(i) >= 0.25 &
               .and. th_fc(i) <= 0.45 .and. th_sat(i) >= 0.40 .and. th_sat(i) <= 0.53 ) then

             if ( k_sat(i) >= 5. .and. k_sat(i) <= 150. ) then
                a_cr(i) = -0.5677 - (k_sat(i) * 4. * 1E-05)
                b_cr(i) = -3.7189 + (0.5922 * log(k_sat(i)))

             else if ( k_sat(i) < 5. ) then
                a_cr(i) = -0.5677 - (5. * 4. * 1E-05)
                b_cr(i) = -3.7189 + (0.5922 * log(5.))

             else if ( k_sat(i) > 150. ) then
                a_cr(i) = -0.5677 - (150. * 4. * 1E-05)
                b_cr(i) = -3.7189 + (0.5922 * log(150.))
                
             end if

          else if ( th_wilt(i) >= 0.20 .and. th_wilt(i) <= 0.42 .and. th_fc(i) >= 0.40 &
               .and. th_fc(i) < 0.58 .and. th_sat(i) >= 0.49 .and. th_sat(i) <= 0.58 ) then

             if ( k_sat(i) >= 1. .and. k_sat(i) <= 150. ) then
                a_cr(i) = -0.6366 + (k_sat(i) * 8. * 1E-04)
                b_cr(i) = -1.9165 + (0.7063 * log(k_sat(i)))

             else if ( k_sat(i) < 1 ) then
                a_cr(i) = -0.6366 + (1. * 8. * 1E-04)
                b_cr(i) = -1.9165 + (0.7063 * log(1.))

             else if ( k_sat(i) > 150 ) then
                a_cr(i) = -0.6366 + (150. * 8. * 1E-04)
                b_cr(i) = -1.9165 + (0.7063 * log(150.))

             end if
             
          end if
       end do
    end if
    
  end subroutine compute_cap_rise_parameters


  subroutine compute_tau( &
       tau, &
       k_sat &
       )

    real(real64), dimension(:), intent(inout) :: tau
    real(real64), dimension(:), intent(in) :: k_sat
    integer(int32) :: n
    integer(int32) :: i
    real(real64) :: tau_i
    
    n = size(tau)
    do i = 1, n
       tau_i = 0.0866 * (k_sat(i) ** 0.35)
       tau_i = real(nint(tau_i * 100.)) / 100.
       if ( tau_i > 1 ) then
          tau(i) = 1.
       else if ( tau_i < 0 ) then
          tau(i) = 0.
       else
          tau(i) = tau_i
       end if
    end do
  end subroutine compute_tau
  
  subroutine compute_th_dry( &
       th_dry, &
       th_wilt &
       )

    real(real64), dimension(:), intent(inout) :: th_dry
    real(real64), dimension(:), intent(in) :: th_wilt
    integer(int32) :: n
    integer(int32) :: i
    n = size(th_dry)
    do i = 1, n
       th_dry(i) = th_wilt(i) / 2.
    end do
    
  end subroutine compute_th_dry
  
  subroutine compute_soil_h_parameters( &
       a_cr, &
       b_cr, &
       tau, &
       th_dry, &
       th_fc, &
       th_sat, &
       th_wilt, &
       k_sat, &
       water_table &
       )

    real(real64), dimension(:), intent(inout) :: a_cr
    real(real64), dimension(:), intent(inout) :: b_cr
    real(real64), dimension(:), intent(inout) :: tau
    real(real64), dimension(:), intent(inout) :: th_dry
    real(real64), dimension(:), intent(in) :: th_fc
    real(real64), dimension(:), intent(in) :: th_sat
    real(real64), dimension(:), intent(in) :: th_wilt
    real(real64), dimension(:), intent(in) :: k_sat
    integer(int32), intent(in) :: water_table
    
    call compute_cap_rise_parameters( &
         a_cr, &
         b_cr, &
         th_fc, &
         th_sat, &
         th_wilt, &
         k_sat, &
         water_table &
         )

    call compute_tau( &
         tau, &
         k_sat &
         )

    call compute_th_dry( &
         th_dry, &
         th_wilt &
         )
    
  end subroutine compute_soil_h_parameters
         
end module soil_hydraulic_parameters
                                        
! %% Calculate drainage characteristic (tau) %%
! % Calculations use equation given by Raes et al. 2012
! for ii = 1:ParamStruct.Soil.nLayer
!     ParamStruct.Soil.Layer.tau(ii) = 0.0866*(ParamStruct.Soil.Layer.Ksat(ii)^0.35);
!     ParamStruct.Soil.Layer.tau(ii) = round((100*ParamStruct.Soil.Layer.tau(ii)))/100;
!     if ParamStruct.Soil.Layer.tau(ii) > 1
!        ParamStruct.Soil.Layer.tau(ii) = 1;
!     elseif ParamStruct.Soil.Layer.tau(ii) < 0
!         ParamStruct.Soil.Layer.tau(ii) = 0;
!     end
! end
    
       

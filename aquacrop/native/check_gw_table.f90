module check_gw_table
  use types
  implicit none

contains

  subroutine update_check_gw_table( &
       th, &
       th_fc_adj, &
       wt_in_soil, &       
       th_s, &
       th_fc, &
       wt, &
       variable_wt, &
       zgw, &
       dz, &
       layer_ix &
       )

    real(real64), dimension(:), intent(inout) :: th
    real(real64), dimension(:), intent(inout) :: th_fc_adj
    integer(int32), intent(inout) :: wt_in_soil
    real(real64), dimension(:), intent(in) :: th_s
    real(real64), dimension(:), intent(in) :: th_fc    
    integer(int32), intent(in) :: wt
    integer(int32), intent(in) :: variable_wt
    real(real64), intent(in) :: zgw
    real(real64), dimension(:), intent(in) :: dz
    integer(int32), dimension(:), intent(in) :: layer_ix
    
    real(real64), allocatable, dimension(:) :: zbot
    real(real64), allocatable, dimension(:) :: ztop
    real(real64), allocatable, dimension(:) :: zmid
    real(real64) :: x_max
    real(real64) :: pf
    real(real64) :: dv
    real(real64) :: dfc
    integer(int32) :: n_comp
    integer(int32) :: i, j, idx, lyri, lyrj, compi
    
    n_comp = size(dz, 1)
    allocate(zbot(n_comp))
    allocate(ztop(n_comp))
    allocate(zmid(n_comp))
    
    if ( wt == 1 .and. variable_wt == 1 ) then
       ! update groundwater conditions for current day
       zbot(1) = dz(1)
       ztop(1) = 0
       zmid(1) = dz(1) / 2
       do i = 2, n_comp
          zbot(i) = zbot(i-1) + dz(i)
          ztop(i) = zbot(i-1)
          zmid(i) = (ztop(i) + zbot(i)) / 2
       end do

       ! check if water table is within modelled soil profile
       if ( zgw >= 0 ) then
          if ( zmid(n_comp) >= zgw ) then
             wt_in_soil = 1
          else
             wt_in_soil = 0
          end if
       end if

       ! if water table is in soil profile, adjust water contents
       if ( wt_in_soil == 1 ) then
          ! find index of first soil compartment which is deeper
          ! than the water table
          compi = 1
          do while ( compi <= n_comp )
             if ( zmid(compi) >= zgw ) then
                idx = compi
             end if
          end do
            
          do i = idx, n_comp
             lyri = layer_ix(i)
             th(i) = th_s(lyri)
          end do
       end if

       ! adjust compartment field capacity, starting in the
       ! bottom compartment
       compi = n_comp
       do while ( compi >= 1 )
          lyri = layer_ix(compi)
          if ( th_fc(lyri) <= 0.1 ) then
             x_max = 1.
          else
             if ( th_fc(lyri) >= 0.3 ) then
                x_max = 2.
             else                      
                pf = 2. + 0.3 * (th_fc(lyri) - 0.1) / 0.2
                x_max = exp(pf * log(10.)) / 100
             end if                   
          end if

          if ( zgw < 0 .or. (zgw - zmid(compi)) >= x_max ) then
             do j = 1, compi
                lyrj = layer_ix(j)
                th_fc_adj(j) = th_fc(lyrj)
             end do
             compi = 0
          else
             if ( th_fc(lyri) >= th_s(lyri) ) then
                th_fc_adj(compi) = th_fc(lyri)
             else
                if ( zmid(compi) >= zgw ) then
                   th_fc_adj(compi) = th_s(lyri)
                else
                   dv = th_s(lyri) - th_fc(lyri)
                   dfc = (dv / (x_max ** 2)) * ((zmid(compi) - (zgw - x_max)) ** 2)
                   th_fc_adj(compi) = th_fc(lyri) + dfc
                end if
             end if
             compi = compi - 1
          end if
       end do
    end if
    
  end subroutine update_check_gw_table

end module check_gw_table

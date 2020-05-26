module inflow
  use types
  use soil_evaporation, only: get_max_comp_idx
  implicit none

contains

  subroutine update_inflow( &
       gw_in, &
       th, &
       water_table, &
       z_gw, &
       th_sat, &
       dz, &
       layer_ix &
       )

    real(real64), intent(inout) :: gw_in
    real(real64), dimension(:), intent(inout) :: th
    integer(int32), intent(in) :: water_table
    real(real64), intent(in) :: z_gw
    real(real64), dimension(:), intent(in) :: th_sat
    real(real64), dimension(:), intent(in) :: dz
    integer(int32), dimension(:), intent(in) :: layer_ix

    integer(int32) :: n_comp
    integer(int32) :: i
    integer(int32) :: max_comp_idx
    integer(int32) :: lyri
    real(real64), allocatable, dimension(:) :: z_bot
    real(real64), allocatable, dimension(:) :: z_mid
    real(real64) :: dth
    
    n_comp = size(dz, 1)
    allocate(z_bot(n_comp))
    allocate(z_mid(n_comp))
    gw_in = 0.
    if ( water_table == 1 ) then
       z_bot(1) = dz(1)
       z_mid(1) = dz(1) / 2.
       do i = 2, n_comp
          z_bot(i) = z_bot(i-1) + dz(i)
          z_mid(i) = (z_bot(i) + z_bot(i-1)) / 2.
       end do

       ! set compartments below water table to saturation
       if ( z_gw <= z_mid(n_comp) ) then
          max_comp_idx = get_max_comp_idx(z_gw, dz, z_mid) ! check this is doing what you think
          do i = max_comp_idx, n_comp
             lyri = layer_ix(i)
             if ( th(i) < th_sat(lyri) ) then
                dth = th_sat(lyri) - th(i)
                th(i) = th_sat(lyri)
                gw_in = gw_in + (dth * 1000. * dz(i))
             end if
          end do
       end if
    end if

    ! clean up
    deallocate(z_bot)
    deallocate(z_mid)
    
  end subroutine update_inflow
  
end module inflow

!     % For compartments below water table, set to saturation %
!     idx = find(zMid >= zGW);
!     for ii = idx:Soil.nComp
!         % Get soil layer
!         layeri = Soil.Comp.Layer(ii);
!         if NewCond.th(ii) < Soil.Layer.th_s(layeri);
!             % Update water content
!             dth = Soil.Layer.th_s(layeri)-NewCond.th(ii);
!             NewCond.th(ii) = Soil.Layer.th_s(layeri);
!             % Update groundwater inflow
!             GwIn = GwIn+(dth*1000*Soil.Comp.dz(ii));
!         end
!     end
! end

! end

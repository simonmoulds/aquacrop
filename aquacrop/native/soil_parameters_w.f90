module soil_parameters_w
  use types
  use soil_parameters
  implicit none

contains

  subroutine compute_soil_parameters_w( &
       cn_bot, &
       cn_top, &
       rew, &
       w_rel, &
       z_germ, &
       cn, &
       th_fc, &
       th_dry, &
       evap_z_surf, &
       z_cn, &
       dz_sum, &
       n_layer, n_cell &
       )
    
    integer(int32), intent(in) :: n_layer, n_cell
    real(real64), dimension(n_cell), intent(inout) :: cn_bot
    real(real64), dimension(n_cell), intent(inout) :: cn_top
    real(real64), dimension(n_cell), intent(inout) :: rew
    real(real64), dimension(n_cell, n_layer), intent(inout) :: w_rel
    real(real64), dimension(n_cell), intent(inout) :: z_germ    
    real(real64), dimension(n_cell), intent(in) :: cn
    real(real64), dimension(n_cell, n_layer), intent(in) :: th_fc
    real(real64), dimension(n_cell, n_layer), intent(in) :: th_dry
    real(real64), dimension(n_cell), intent(in) :: evap_z_surf
    real(real64), dimension(n_cell), intent(in) :: z_cn
    real(real64), dimension(n_layer), intent(in) :: dz_sum
    integer(int32) :: i
    
    do i = 1, n_cell
       call compute_soil_parameters( &
            cn_bot(i), &
            cn_top(i), &
            rew(i), &
            w_rel(i,:), &
            z_germ(i), &
            cn(i), &
            th_fc(i,:), &
            th_dry(i,:), &
            evap_z_surf(i), &
            z_cn(i), &
            dz_sum(:) &
            )
    end do

  end subroutine compute_soil_parameters_w
  
end module soil_parameters_w

       

    

module soil_hydraulic_parameters_w
  use types
  use soil_hydraulic_parameters
  implicit none

contains
  
  subroutine compute_soil_h_parameters_w( &
       a_cr, &
       b_cr, &
       tau, &
       th_dry, &
       th_fc, &
       th_sat, &
       th_wilt, &
       k_sat, &
       water_table, &
       n_layer, n_cell &
       )

    integer(int32), intent(in) :: n_layer, n_cell
    real(real64), dimension(n_cell, n_layer), intent(inout) :: a_cr
    real(real64), dimension(n_cell, n_layer), intent(inout) :: b_cr
    real(real64), dimension(n_cell, n_layer), intent(inout) :: tau
    real(real64), dimension(n_cell, n_layer), intent(inout) :: th_dry
    real(real64), dimension(n_cell, n_layer), intent(in) :: th_fc
    real(real64), dimension(n_cell, n_layer), intent(in) :: th_sat
    real(real64), dimension(n_cell, n_layer), intent(in) :: th_wilt
    real(real64), dimension(n_cell, n_layer), intent(in) :: k_sat
    integer(int32), intent(in) :: water_table
    integer(int32) :: i

    do i = 1, n_cell
       call compute_soil_h_parameters( &
            a_cr(i,:), &
            b_cr(i,:), &
            tau(i,:), &
            th_dry(i,:), &
            th_fc(i,:), &
            th_sat(i,:), &
            th_wilt(i,:), &
            k_sat(i,:), &
            water_table &
            )
    end do
    
  end subroutine compute_soil_h_parameters_w
  
end module soil_hydraulic_parameters_w

       
    

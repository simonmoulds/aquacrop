module gdd_w
  use types
  use gdd, only: update_gdd
  implicit none

contains

  subroutine update_gdd_w( &
       gdd, &
       gdd_cum, &
       gdd_method, &
       t_max, &
       t_min, &
       t_base, &
       t_upp, &
       growing_season, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_crop, n_farm, n_cell), intent(inout) :: gdd
    real(real64), dimension(n_crop, n_farm, n_cell), intent(inout) :: gdd_cum
    integer(int32), intent(in) :: gdd_method
    real(real64), dimension(n_crop, n_farm, n_cell), intent(in) :: t_max
    real(real64), dimension(n_crop, n_farm, n_cell), intent(in) :: t_min
    real(real64), dimension(n_crop, n_farm, n_cell), intent(in) :: t_base
    real(real64), dimension(n_crop, n_farm, n_cell), intent(in) :: t_upp
    integer(int32), dimension(n_crop, n_farm, n_cell), intent(in) :: growing_season
    integer(int32) :: i, j, k
    
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_gdd( &
                  gdd(i,j,k), &
                  gdd_cum(i,j,k), &
                  gdd_method, &
                  t_max(i,j,k), &
                  t_min(i,j,k), &
                  t_base(i,j,k), &
                  t_upp(i,j,k), &
                  growing_season(i,j,k) &
                  )
          end do
       end do
    end do
    
  end subroutine update_gdd_w
  
end module gdd_w

                  
    
       

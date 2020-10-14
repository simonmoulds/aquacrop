module crop_yield_w
  use types
  use crop_yield, only: update_crop_yield
  implicit none

contains

  subroutine update_crop_yield_w( &
       yield, &
       crop_mature, &
       maturity, &
       b, &
       hi_adj, &
       gdd_cum, &
       growing_season, &
       growing_season_day1, &
       calendar_type, &
       dap, &       
       delayed_cds, &
       delayed_gdds, &
       n_crop, n_farm, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    integer(int32), intent(in) :: calendar_type
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: yield
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: crop_mature
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: maturity
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: b
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: hi_adj
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: gdd_cum
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season_day1
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: dap
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_cds
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_gdds

    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_crop_yield( &
                  yield(k,j,i), &
                  crop_mature(k,j,i), &
                  maturity(k,j,i), &
                  b(k,j,i), &
                  hi_adj(k,j,i), &
                  gdd_cum(k,j,i), &
                  growing_season(k,j,i), &
                  growing_season_day1(k,j,i), &
                  calendar_type, &
                  dap(k,j,i), &       
                  delayed_cds(k,j,i), &
                  delayed_gdds(k,j,i) &
                  )
          end do
       end do
    end do
    
  end subroutine update_crop_yield_w
  
end module crop_yield_w

    

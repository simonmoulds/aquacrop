module growth_stage_w
  use types
  use growth_stage, only: update_growth_stage
  implicit none

contains

  subroutine update_growth_stage_w( &
       growth_stage, &
       canopy_10pct, &
       max_canopy, &
       senescence, &
       gdd_cum, &
       dap, &
       delayed_cds, &
       delayed_gdds, &       
       calendar_type, &
       growing_season, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: growth_stage
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: canopy_10pct
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: max_canopy
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: senescence
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: gdd_cum
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: dap
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_cds
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_gdds
    integer(int32), intent(in) :: calendar_type
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season
    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_growth_stage( &
                  growth_stage(i,j,k), &
                  canopy_10pct(i,j,k), &
                  max_canopy(i,j,k), &
                  senescence(i,j,k), &
                  gdd_cum(i,j,k), &
                  dap(i,j,k), &
                  delayed_cds(i,j,k), &
                  delayed_gdds(i,j,k), &       
                  calendar_type, &
                  growing_season(i,j,k) &
                  )
          end do
       end do
    end do
    
  end subroutine update_growth_stage_w
  
end module growth_stage_w

                  

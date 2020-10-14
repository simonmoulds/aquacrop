module root_dev_w
  use types
  use root_dev, only: update_root_dev
  implicit none
  
contains

  subroutine update_root_dev_w( &
       z_root, &
       r_cor, &
       z_min, &
       z_max, &
       pct_z_min, &
       emergence, &
       max_rooting, &
       fshape_r, &
       fshape_ex, &
       sx_bot, &
       sx_top, &
       dap, &
       gdd, &
       gdd_cum, &
       delayed_cds, &
       delayed_gdds, &
       tr_ratio, &
       germination, &
       z_res, &
       water_table, &
       z_gw, &
       calendar_type, &
       growing_season_day1, &
       growing_season, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: z_root
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: r_cor
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_min
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_max
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: pct_z_min
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: emergence
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: max_rooting
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: fshape_r
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: fshape_ex
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: sx_bot
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: sx_top
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: dap
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: gdd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: gdd_cum
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_cds
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_gdds
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: tr_ratio
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: germination
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: z_res
    integer(int32), intent(in) :: water_table
    real(real64), dimension(n_cell), intent(in) :: z_gw
    integer(int32), intent(in) :: calendar_type
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season_day1
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season
    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_root_dev( &
                  z_root(k,j,i), &
                  r_cor(k,j,i), &
                  z_min(k,j,i), &
                  z_max(k,j,i), &
                  pct_z_min(k,j,i), &
                  emergence(k,j,i), &
                  max_rooting(k,j,i), &
                  fshape_r(k,j,i), &
                  fshape_ex(k,j,i), &
                  sx_bot(k,j,i), &
                  sx_top(k,j,i), &
                  dap(k,j,i), &
                  gdd(k,j,i), &
                  gdd_cum(k,j,i), &
                  delayed_cds(k,j,i), &
                  delayed_gdds(k,j,i), &
                  tr_ratio(k,j,i), &
                  germination(k,j,i), &
                  z_res(k,j,i), &
                  water_table, &
                  z_gw(k), &
                  calendar_type, &
                  growing_season_day1(k,j,i), &
                  growing_season(k,j,i) &
                  )
          end do
       end do
    end do
  end subroutine update_root_dev_w
end module root_dev_w

                  
  

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
       growing_season, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: z_root
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: r_cor
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_min
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_max
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: pct_z_min
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: emergence
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: max_rooting
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: fshape_r
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: fshape_ex
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: sx_bot
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: sx_top
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: dap
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: gdd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: gdd_cum
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_cds
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_gdds
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: tr_ratio
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: germination
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: z_res
    integer(int32), intent(in) :: water_table
    real(real64), dimension(n_cell), intent(in) :: z_gw
    integer(int32), intent(in) :: calendar_type
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season
    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_root_dev( &
                  z_root(i,j,k), &
                  r_cor(i,j,k), &
                  z_min(i,j,k), &
                  z_max(i,j,k), &
                  pct_z_min(i,j,k), &
                  emergence(i,j,k), &
                  max_rooting(i,j,k), &
                  fshape_r(i,j,k), &
                  fshape_ex(i,j,k), &
                  sx_bot(i,j,k), &
                  sx_top(i,j,k), &
                  dap(i,j,k), &
                  gdd(i,j,k), &
                  gdd_cum(i,j,k), &
                  delayed_cds(i,j,k), &
                  delayed_gdds(i,j,k), &
                  tr_ratio(i,j,k), &
                  germination(i,j,k), &
                  z_res(i,j,k), &
                  water_table, &
                  z_gw(k), &
                  calendar_type, &
                  growing_season(i,j,k) &
                  )
          end do
       end do
    end do
  end subroutine update_root_dev_w
end module root_dev_w

                  
  

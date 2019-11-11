module capillary_rise_w
  use types
  use capillary_rise, only: update_cap_rise
  implicit none

contains

  subroutine update_cap_rise_w( &
       cr_tot, &
       th, &
       th_wp, &
       th_fc, &
       th_fc_adj, &
       k_sat, &
       a_cr, &
       b_cr, &
       f_shape_cr, &
       flux_out, &
       water_table, &
       zgw, &
       dz, &
       dz_layer, &
       layer_ix, &
       n_farm, n_crop, n_comp, n_layer, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cr_tot
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: th
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_wp
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_fc
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(in) :: th_fc_adj
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: k_sat
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: a_cr
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: b_cr
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_shape_cr
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(in) :: flux_out
    integer(int32), intent(in) :: water_table
    real(real64), dimension(n_cell), intent(in) :: zgw
    real(real64), dimension(n_comp), intent(in) :: dz
    real(real64), dimension(n_layer), intent(in) :: dz_layer
    integer(int32), dimension(n_comp), intent(in) :: layer_ix

    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_cap_rise( &
                  cr_tot(k,j,i), &
                  th(k,:,j,i), &
                  th_wp(k,:,j,i), &
                  th_fc(k,:,j,i), &
                  th_fc_adj(k,:,j,i), &
                  k_sat(k,:,j,i), &
                  a_cr(k,:,j,i), &
                  b_cr(k,:,j,i), &
                  f_shape_cr(k,j,i), &
                  flux_out(k,:,j,i), &
                  water_table, &
                  zgw(k), &
                  dz, &
                  dz_layer, &
                  layer_ix &
                  )
          end do
       end do
    end do
  end subroutine update_cap_rise_w
  
  ! subroutine update_cap_rise_w( &
  !      cr_tot, &
  !      th, &
  !      th_wp, &
  !      th_fc, &
  !      th_fc_adj, &
  !      k_sat, &
  !      a_cr, &
  !      b_cr, &
  !      f_shape_cr, &
  !      flux_out, &
  !      water_table, &
  !      zgw, &
  !      dz, &
  !      dz_layer, &
  !      layer_ix, &
  !      n_farm, n_crop, n_comp, n_layer, n_cell &
  !      )

  !   integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cr_tot
  !   real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: th
  !   real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_wp
  !   real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_fc
  !   real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(in) :: th_fc_adj
  !   real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: k_sat
  !   real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: a_cr
  !   real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: b_cr
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_cr
  !   real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(in) :: flux_out
  !   integer(int32), intent(in) :: water_table
  !   real(real64), dimension(n_cell), intent(in) :: zgw
  !   real(real64), dimension(n_comp), intent(in) :: dz
  !   real(real64), dimension(n_layer), intent(in) :: dz_layer
  !   integer(int32), dimension(n_comp), intent(in) :: layer_ix

  !   integer(int32) :: i, j, k
  !   do i = 1, n_farm
  !      do j = 1, n_crop
  !         do k = 1, n_cell
  !            call update_cap_rise( &
  !                 cr_tot(i,j,k), &
  !                 th(i,j,:,k), &
  !                 th_wp(i,j,:,k), &
  !                 th_fc(i,j,:,k), &
  !                 th_fc_adj(i,j,:,k), &
  !                 k_sat(i,j,:,k), &
  !                 a_cr(i,j,:,k), &
  !                 b_cr(i,j,:,k), &
  !                 f_shape_cr(i,j,k), &
  !                 flux_out(i,j,:,k), &
  !                 water_table, &
  !                 zgw(k), &
  !                 dz, &
  !                 dz_layer, &
  !                 layer_ix &
  !                 )
  !         end do
  !      end do
  !   end do
  ! end subroutine update_cap_rise_w
  
end module capillary_rise_w


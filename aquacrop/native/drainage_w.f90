module drainage_w
  use types
  use drainage, only: update_drainage
  implicit none

contains

  subroutine update_drainage_w( &
       th, &
       deep_perc, &
       flux_out, &
       th_sat, &
       th_fc, &
       k_sat, &
       tau, &
       th_fc_adj, &
       dz, &
       dzsum, &
       layer_ix, &
       n_farm, n_crop, n_comp, n_layer, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: th
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: deep_perc
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(inout) :: flux_out
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_sat
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: th_fc
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: k_sat
    real(real64), dimension(n_cell, n_layer, n_crop, n_farm), intent(in) :: tau
    real(real64), dimension(n_cell, n_comp, n_crop, n_farm), intent(in) :: th_fc_adj
    real(real64), dimension(n_comp), intent(in) :: dz
    real(real64), dimension(n_comp), intent(in) :: dzsum
    integer(int32), dimension(n_comp), intent(in) :: layer_ix
    integer(int32) :: i, j, k
    
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_drainage( &
                  th(k,:,j,i), &
                  deep_perc(k,j,i), &
                  flux_out(k,:,j,i), &
                  th_sat(k,:,j,i), &
                  th_fc(k,:,j,i), &
                  k_sat(k,:,j,i), &
                  tau(k,:,j,i), &
                  th_fc_adj(k,:,j,i), &
                  dz, &
                  dzsum, &
                  layer_ix &
                  )
             
          end do
       end do
    end do
  end subroutine update_drainage_w
  
  ! subroutine update_drainage_w( &
  !      th, &
  !      deep_perc, &
  !      flux_out, &
  !      th_sat, &
  !      th_fc, &
  !      k_sat, &
  !      tau, &
  !      th_fc_adj, &
  !      dz, &
  !      dzsum, &
  !      layer_ix, &
  !      n_farm, n_crop, n_comp, n_layer, n_cell &
  !      )

  !   integer(int32), intent(in) :: n_farm, n_crop, n_comp, n_layer, n_cell
  !   real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: th
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: deep_perc
  !   real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(inout) :: flux_out
  !   real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_sat
  !   real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: th_fc
  !   real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: k_sat
  !   real(real64), dimension(n_farm, n_crop, n_layer, n_cell), intent(in) :: tau
  !   real(real64), dimension(n_farm, n_crop, n_comp, n_cell), intent(in) :: th_fc_adj
  !   real(real64), dimension(n_comp), intent(in) :: dz
  !   real(real64), dimension(n_comp), intent(in) :: dzsum
  !   integer(int32), dimension(n_comp), intent(in) :: layer_ix
  !   integer(int32) :: i, j, k
    
  !   do i = 1, n_farm
  !      do j = 1, n_crop
  !         do k = 1, n_cell
  !            call update_drainage( &
  !                 th(i,j,:,k), &
  !                 deep_perc(i,j,k), &
  !                 flux_out(i,j,:,k), &
  !                 th_sat(i,j,:,k), &
  !                 th_fc(i,j,:,k), &
  !                 k_sat(i,j,:,k), &
  !                 tau(i,j,:,k), &
  !                 th_fc_adj(i,j,:,k), &
  !                 dz, &
  !                 dzsum, &
  !                 layer_ix &
  !                 )
             
  !         end do
  !      end do
  !   end do
  ! end subroutine update_drainage_w
  
end module drainage_w

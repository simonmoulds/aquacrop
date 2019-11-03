module canopy_cover_w
  use types
  use canopy_cover, only: update_canopy_cover
  implicit none

contains


  subroutine update_canopy_cover_w( &
       cc, &
       cc_prev, &       
       cc_adj, &
       cc_ns, &
       cc_adj_ns, &
       ccx_w, &
       ccx_act, &
       ccx_w_ns, &
       ccx_act_ns, &       
       cc0_adj, &
       ccx_early_sen, &       
       t_early_sen, &
       premat_senes, &
       crop_dead, &
       gdd, &
       gddcum, &       
       cc0, &
       ccx, &
       cgc, &
       cdc, &
       emergence, &
       maturity, &
       senescence, &
       canopy_dev_end, &
       dr, &
       taw, &
       et_ref, &
       et_adj, &
       p_up1, &
       p_up2, &
       p_up3, &
       p_up4, &
       p_lo1, &
       p_lo2, &
       p_lo3, &
       p_lo4, &
       f_shape_w1, &
       f_shape_w2, &
       f_shape_w3, &
       f_shape_w4, &
       growing_season, &
       calendar_type, &
       dap, &
       delayed_cds, &
       delayed_gdds, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    integer(int32), intent(in) :: calendar_type
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_prev
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_adj
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc_adj_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_w
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_act
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_w_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_act_ns
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: cc0_adj
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: ccx_early_sen
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(inout) :: t_early_sen
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: premat_senes
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(inout) :: crop_dead

    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: gdd
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: gddcum
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cc0
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: ccx
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cgc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: cdc
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: emergence
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: maturity
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: senescence
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: canopy_dev_end
    
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: dr
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: taw
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: et_ref
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: et_adj
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_up1
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_up2
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_up3
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_up4
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_lo1
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_lo2
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_lo3
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: p_lo4
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_shape_w1
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_shape_w2
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_shape_w3
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: f_shape_w4    
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: growing_season
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: dap
    integer(int32), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_cds
    real(real64), dimension(n_cell, n_crop, n_farm), intent(in) :: delayed_gdds

    integer(int32) :: i, j, k

    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell             
             call update_canopy_cover( &
                  cc(k,j,i), &
                  cc_prev(k,j,i), &
                  cc_adj(k,j,i), &
                  cc_ns(k,j,i), &
                  cc_adj_ns(k,j,i), &
                  ccx_w(k,j,i), &
                  ccx_act(k,j,i), &
                  ccx_w_ns(k,j,i), &
                  ccx_act_ns(k,j,i), &       
                  cc0_adj(k,j,i), &
                  ccx_early_sen(k,j,i), &       
                  t_early_sen(k,j,i), &
                  premat_senes(k,j,i), &
                  crop_dead(k,j,i), &
                  gdd(k,j,i), &
                  gddcum(k,j,i), &       
                  cc0(k,j,i), &
                  ccx(k,j,i), &
                  cgc(k,j,i), &
                  cdc(k,j,i), &
                  emergence(k,j,i), &
                  maturity(k,j,i), &
                  senescence(k,j,i), &
                  canopy_dev_end(k,j,i), &
                  dr(k,j,i), &
                  taw(k,j,i), &
                  et_ref(k,j,i), &
                  et_adj(k,j,i), &
                  p_up1(k,j,i), &
                  p_up2(k,j,i), &
                  p_up3(k,j,i), &
                  p_up4(k,j,i), &
                  p_lo1(k,j,i), &
                  p_lo2(k,j,i), &
                  p_lo3(k,j,i), &
                  p_lo4(k,j,i), &
                  f_shape_w1(k,j,i), &
                  f_shape_w2(k,j,i), &
                  f_shape_w3(k,j,i), &
                  f_shape_w4(k,j,i), &
                  growing_season(k,j,i), &
                  calendar_type, &
                  dap(k,j,i), &
                  delayed_cds(k,j,i), &
                  delayed_gdds(k,j,i) &
                  )
             
          end do
       end do
    end do
    
  end subroutine update_canopy_cover_w
    
  ! subroutine update_canopy_cover_w( &
  !      cc, &
  !      cc_prev, &       
  !      cc_adj, &
  !      cc_ns, &
  !      cc_adj_ns, &
  !      ccx_w, &
  !      ccx_act, &
  !      ccx_w_ns, &
  !      ccx_act_ns, &       
  !      cc0_adj, &
  !      ccx_early_sen, &       
  !      t_early_sen, &
  !      premat_senes, &
  !      crop_dead, &
  !      gdd, &
  !      gddcum, &       
  !      cc0, &
  !      ccx, &
  !      cgc, &
  !      cdc, &
  !      emergence, &
  !      maturity, &
  !      senescence, &
  !      canopy_dev_end, &
  !      dr, &
  !      taw, &
  !      et_ref, &
  !      et_adj, &
  !      p_up1, &
  !      p_up2, &
  !      p_up3, &
  !      p_up4, &
  !      p_lo1, &
  !      p_lo2, &
  !      p_lo3, &
  !      p_lo4, &
  !      f_shape_w1, &
  !      f_shape_w2, &
  !      f_shape_w3, &
  !      f_shape_w4, &
  !      growing_season, &
  !      calendar_type, &
  !      dap, &
  !      delayed_cds, &
  !      delayed_gdds, &
  !      n_farm, n_crop, n_cell &
  !      )

  !   integer(int32), intent(in) :: n_farm, n_crop, n_cell
  !   integer(int32), intent(in) :: calendar_type
    
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc_prev
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc_adj
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc_ns
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc_adj_ns
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ccx_w
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ccx_act
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ccx_w_ns
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ccx_act_ns
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: cc0_adj
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: ccx_early_sen
    
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: t_early_sen
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: premat_senes
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(inout) :: crop_dead

  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: gdd
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: gddcum
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cc0
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: ccx
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cgc
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: cdc
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: emergence
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: maturity
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: senescence
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: canopy_dev_end
    
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: dr
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: taw
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: et_ref
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: et_adj
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up1
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up2
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up3
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_up4
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo1
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo2
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo3
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: p_lo4
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w1
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w2
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w3
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shape_w4    
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: dap
  !   integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_cds
  !   real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_gdds

  !   integer(int32) :: i, j, k

  !   do i = 1, n_farm
  !      do j = 1, n_crop
  !         do k = 1, n_cell             
  !            call update_canopy_cover( &
  !                 cc(i,j,k), &
  !                 cc_prev(i,j,k), &
  !                 cc_adj(i,j,k), &
  !                 cc_ns(i,j,k), &
  !                 cc_adj_ns(i,j,k), &
  !                 ccx_w(i,j,k), &
  !                 ccx_act(i,j,k), &
  !                 ccx_w_ns(i,j,k), &
  !                 ccx_act_ns(i,j,k), &       
  !                 cc0_adj(i,j,k), &
  !                 ccx_early_sen(i,j,k), &       
  !                 t_early_sen(i,j,k), &
  !                 premat_senes(i,j,k), &
  !                 crop_dead(i,j,k), &
  !                 gdd(i,j,k), &
  !                 gddcum(i,j,k), &       
  !                 cc0(i,j,k), &
  !                 ccx(i,j,k), &
  !                 cgc(i,j,k), &
  !                 cdc(i,j,k), &
  !                 emergence(i,j,k), &
  !                 maturity(i,j,k), &
  !                 senescence(i,j,k), &
  !                 canopy_dev_end(i,j,k), &
  !                 dr(i,j,k), &
  !                 taw(i,j,k), &
  !                 et_ref(i,j,k), &
  !                 et_adj(i,j,k), &
  !                 p_up1(i,j,k), &
  !                 p_up2(i,j,k), &
  !                 p_up3(i,j,k), &
  !                 p_up4(i,j,k), &
  !                 p_lo1(i,j,k), &
  !                 p_lo2(i,j,k), &
  !                 p_lo3(i,j,k), &
  !                 p_lo4(i,j,k), &
  !                 f_shape_w1(i,j,k), &
  !                 f_shape_w2(i,j,k), &
  !                 f_shape_w3(i,j,k), &
  !                 f_shape_w4(i,j,k), &
  !                 growing_season(i,j,k), &
  !                 calendar_type, &
  !                 dap(i,j,k), &
  !                 delayed_cds(i,j,k), &
  !                 delayed_gdds(i,j,k) &
  !                 )
             
  !         end do
  !      end do
  !   end do
    
  ! end subroutine update_canopy_cover_w
  
end module canopy_cover_w

             

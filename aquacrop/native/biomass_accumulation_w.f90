module biomass_accumulation_w
  use types
  use biomass_accumulation, only: update_biomass_accum
  implicit none

contains

  subroutine update_biomass_accum_w( &
       et_ref, &
       tr, &
       tr_pot, &
       b, &
       b_ns, &
       bio_temp_stress, &
       gdd, &
       gdd_up, &
       gdd_lo, &
       pol_heat_stress, &
       t_max, &
       t_max_up, &
       t_max_lo, &
       f_shp_b, &
       pol_cold_stress, &
       t_min, &
       t_min_up, &
       t_min_lo, &       
       hi_ref, &
       pct_lag_phase, &
       yld_form_cd, &
       wp, &
       wpy, &
       f_co2, &
       hi_start_cd, &
       delayed_cds, &
       dap, &
       crop_type, &
       determinant, &
       growing_season, &
       n_farm, n_crop, n_cell &
       )

    integer(int32), intent(in) :: n_farm, n_crop, n_cell
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: et_ref
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: tr
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: tr_pot
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: b
    real(real64), dimension(n_farm, n_crop, n_cell), intent(inout) :: b_ns
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: bio_temp_stress    
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: gdd, gdd_up, gdd_lo
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: pol_heat_stress
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_max, t_max_up, t_max_lo
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: pol_cold_stress
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: t_min, t_min_up, t_min_lo
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_shp_b
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_ref
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: pct_lag_phase
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: yld_form_cd
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: wp
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: wpy
    real(real64), dimension(n_farm, n_crop, n_cell), intent(in) :: f_co2
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: hi_start_cd
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: delayed_cds
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: dap
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: crop_type
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: determinant        
    integer(int32), dimension(n_farm, n_crop, n_cell), intent(in) :: growing_season

    integer(int32) :: i, j, k
    do i = 1, n_farm
       do j = 1, n_crop
          do k = 1, n_cell
             call update_biomass_accum( &
                  et_ref(i,j,k), &
                  tr(i,j,k), &
                  tr_pot(i,j,k), &
                  b(i,j,k), &
                  b_ns(i,j,k), &
                  bio_temp_stress(i,j,k), &
                  gdd(i,j,k), &
                  gdd_up(i,j,k), &
                  gdd_lo(i,j,k), &
                  pol_heat_stress(i,j,k), &
                  t_max(i,j,k), &
                  t_max_up(i,j,k), &
                  t_max_lo(i,j,k), &
                  f_shp_b(i,j,k), &
                  pol_cold_stress(i,j,k), &
                  t_min(i,j,k), &
                  t_min_up(i,j,k), &
                  t_min_lo(i,j,k), &       
                  hi_ref(i,j,k), &
                  pct_lag_phase(i,j,k), &
                  yld_form_cd(i,j,k), &
                  wp(i,j,k), &
                  wpy(i,j,k), &
                  f_co2(i,j,k), &
                  hi_start_cd(i,j,k), &
                  delayed_cds(i,j,k), &
                  dap(i,j,k), &
                  crop_type(i,j,k), &
                  determinant(i,j,k), &
                  growing_season(i,j,k) &
                  )
          end do
       end do
    end do
    
  end subroutine update_biomass_accum_w
  
end module biomass_accumulation_w

                  
    

module recursion_manager_mod
  use iso_c_binding
  use recursion_plugin_mod
  implicit none
  public :: init_recursion, run_recursion, finalize_recursion
  type(c_ptr) :: rec_ptr = c_null_ptr
contains
  subroutine init_recursion(kk, nmax, lld, nhall, nee, ntype, psi_b, hall, ee, lsham, izero, iz)
    ! Input dimensions and host arrays.
    integer, intent(in) :: kk, nmax, lld, nhall, nee, ntype
    complex(c_double_complex), intent(in), target :: psi_b(*)
    complex(c_double_complex), intent(in), target :: hall(*)
    complex(c_double_complex), intent(in), target :: ee(*)
    complex(c_double_complex), intent(in), target :: lsham(*)
    integer, intent(in), target :: izero(*)
    integer, intent(in), target :: iz(*)
    ! Create device structure with extra dimensions.
    rec_ptr = create_recursion(kk, nmax, lld, nhall, nee, ntype)
    ! Copy host arrays into device memory.
    call copy_psi_b_to_device(rec_ptr, c_loc(psi_b(1)))
    call copy_hall_to_device(rec_ptr, c_loc(hall(1)))
    call copy_ee_to_device(rec_ptr, c_loc(ee(1)))
    call copy_lsham_to_device(rec_ptr, c_loc(lsham(1)))
    call copy_izero_to_host(rec_ptr, c_loc(izero(1)))
    call copy_iz_to_device(rec_ptr, c_loc(iz(1)))
  end subroutine init_recursion

  subroutine run_recursion()
    ! Call the CUDA recursion step.
    call run_recursion_step(rec_ptr)
  end subroutine run_recursion

  subroutine finalize_recursion(psi_b_result, atemp_b_result, b2temp_b_result)
    ! Copy out updated device arrays into host arrays.
    complex(c_double_complex), intent(out), target :: psi_b_result(*)
    complex(c_double_complex), intent(out), target :: atemp_b_result(*)
    complex(c_double_complex), intent(out), target :: b2temp_b_result(*)
    call copy_psi_b_to_host(rec_ptr, c_loc(psi_b_result(1)))
    call copy_atemp_b_to_host(rec_ptr, c_loc(atemp_b_result(1)))
    call copy_b2temp_b_to_host(rec_ptr, c_loc(b2temp_b_result(1)))
    ! Destroy the device structure.
    call destroy_recursion(rec_ptr)
    rec_ptr = c_null_ptr
  end subroutine finalize_recursion
end module recursion_manager_mod

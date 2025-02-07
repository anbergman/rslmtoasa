module recursion_manager_mod
  use precision_mod
  use iso_c_binding
  use recursion_plugin_mod
  implicit none
  private
  public :: init_recursion, run_recursion, finalize_recursion

  ! An opaque pointer to our C/CUDA Recursion structure.
  type(c_ptr) :: rec_ptr = c_null_ptr

contains

  !---------------------------------------------------------------------------
  ! init_recursion
  !
  ! Given that your Fortran code has allocated and populated all needed arrays,
  ! this routine creates the device-side Recursion structure and copies the Fortran
  ! arrays into device memory.
  !
  ! Input:
  !   kk, nmax, lld       - dimensions (total atoms, impurity atoms, recursion steps)
  !   psi_b, hall, ee, lsham, izero
  !         - Fortran arrays (declared with the TARGET attribute) allocated contiguously.
  !---------------------------------------------------------------------------
  subroutine init_recursion(kk, nmax, lld, psi_b, hall, ee, lsham, izero)
    integer, intent(in) :: kk, nmax, lld
    complex(rp), intent(in), target :: psi_b(*)
    complex(rp), intent(in), target :: hall(*)
    complex(rp), intent(in), target :: ee(*)
    complex(rp), intent(in), target :: lsham(*)
    integer, intent(in), target :: izero(*)
    ! Create the device structure.
    rec_ptr = create_recursion(kk, nmax, lld)
    ! Copy the Fortran arrays into device memory.
    call copy_psi_b_to_device(rec_ptr, c_loc(psi_b(1)))
    call copy_hall_to_device(rec_ptr, c_loc(hall(1)))
    call copy_ee_to_device(rec_ptr, c_loc(ee(1)))
    call copy_lsham_to_device(rec_ptr, c_loc(lsham(1)))
    call copy_izero_to_host(rec_ptr, c_loc(izero(1)))
  end subroutine init_recursion

  !---------------------------------------------------------------------------
  ! run_recursion
  !
  ! This routine wraps the call to the GPU–accelerated recursion step.
  ! Replace your original call to this%crecal_b() with a call to run_recursion().
  !---------------------------------------------------------------------------
  subroutine run_recursion()
    call run_recursion_step(rec_ptr)
  end subroutine run_recursion

  !---------------------------------------------------------------------------
  ! finalize_recursion
  !
  ! After you have run the recursion step (or several steps), you can call this
  ! routine to copy the results back to host memory. In this example we copy out:
  !   - psi_b (which might have been updated by the recursion step)
  !   - atemp_b (which in the plugin corresponds to the summation matrix)
  !
  ! The host arrays psi_b_result and atemp_b_result must be allocated with the
  ! TARGET attribute.
  !---------------------------------------------------------------------------
  subroutine finalize_recursion(psi_b_result, atemp_b_result, b2temp_b_result)
    complex(c_double_complex), intent(out), target :: psi_b_result(*)
    complex(c_double_complex), intent(out), target :: atemp_b_result(*)
    complex(c_double_complex), intent(out), target :: b2temp_b_result(*)
    call copy_psi_b_to_host(rec_ptr, c_loc(psi_b_result(1)))
    call copy_atemp_b_to_host(rec_ptr, c_loc(atemp_b_result(1)))
    call copy_b2temp_b_to_host(rec_ptr, c_loc(b2temp_b_result(1)))
    call destroy_recursion(rec_ptr)
    rec_ptr = c_null_ptr
  end subroutine finalize_recursion

end module recursion_manager_mod


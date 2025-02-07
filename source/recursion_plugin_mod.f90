module recursion_plugin_mod
  use iso_c_binding
  implicit none
  private
  public :: create_recursion, destroy_recursion
  public :: copy_psi_b_to_device, copy_hall_to_device, copy_ee_to_device, copy_lsham_to_device, copy_izero_to_host
  public :: run_recursion_step, copy_psi_b_to_host, copy_atemp_b_to_host
  interface
    function create_recursion(kk, nmax, lld) bind(C, name="create_recursion")
      import :: c_int, c_ptr
      integer(c_int), value :: kk, nmax, lld
      type(c_ptr) :: create_recursion
    end function create_recursion

    subroutine destroy_recursion(rec) bind(C, name="destroy_recursion")
      import :: c_ptr
      type(c_ptr), value :: rec
    end subroutine destroy_recursion

    subroutine copy_psi_b_to_device(rec, host_psi_b) bind(C, name="copy_psi_b_to_device")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_psi_b
    end subroutine copy_psi_b_to_device

    subroutine copy_hall_to_device(rec, host_hall) bind(C, name="copy_hall_to_device")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_hall
    end subroutine copy_hall_to_device

    subroutine copy_ee_to_device(rec, host_ee) bind(C, name="copy_ee_to_device")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_ee
    end subroutine copy_ee_to_device

    subroutine copy_lsham_to_device(rec, host_lsham) bind(C, name="copy_lsham_to_device_")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_lsham
    end subroutine copy_lsham_to_device

    subroutine copy_izero_to_host(rec, host_izero) bind(C, name="copy_izero_to_host")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_izero
    end subroutine copy_izero_to_host

    subroutine run_recursion_step(rec) bind(C, name="run_recursion_step")
      import :: c_ptr
      type(c_ptr), value :: rec
    end subroutine run_recursion_step

    subroutine copy_psi_b_to_host(rec, host_psi_b) bind(C, name="copy_psi_b_to_host")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_psi_b
    end subroutine copy_psi_b_to_host

    subroutine copy_atemp_b_to_host(rec, host_atemp_b) bind(C, name="copy_atemp_b_to_host_")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_atemp_b
    end subroutine copy_atemp_b_to_host

    subroutine copy_b2temp_b_to_host(rec, host_b2temp_b) bind(C, name="copy_b2temp_b_to_host_")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_b2temp_b
    end subroutine copy_b2temp_b_to_host
  end interface
end module recursion_plugin_mod

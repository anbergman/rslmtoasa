module recursion_plugin_mod
  use iso_c_binding
  implicit none
  interface
    function create_recursion(kk, nmax, lld) bind(C, name="create_recursion_")
      import :: c_int, c_ptr
      integer(c_int), value :: kk, nmax, lld
      type(c_ptr) :: create_recursion
    end function create_recursion

    subroutine destroy_recursion(rec) bind(C, name="destroy_recursion_")
      import :: c_ptr
      type(c_ptr), value :: rec
    end subroutine destroy_recursion

    subroutine copy_psi_b_to_device(rec, host_psi_b) bind(C, name="copy_psi_b_to_device_")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_psi_b
    end subroutine copy_psi_b_to_device

    subroutine copy_ee_to_device(rec, host_ee) bind(C, name="copy_ee_to_device_")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_ee
    end subroutine copy_ee_to_device

    subroutine copy_hall_to_device(rec, host_hall) bind(C, name="copy_hall_to_device_")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_hall
    end subroutine copy_hall_to_device

    ! (Similarly, declare copy_hall_to_device, copy_ee_to_device, etc.)
    
    subroutine run_recursion_step(rec) bind(C, name="run_recursion_step_")
      import :: c_ptr
      type(c_ptr), value :: rec
    end subroutine run_recursion_step

    subroutine copy_psi_b_to_host(rec, host_psi_b) bind(C, name="copy_psi_b_to_host_")
      import :: c_ptr
      type(c_ptr), value :: rec
      type(c_ptr), value :: host_psi_b
    end subroutine copy_psi_b_to_host
  end interface
end module recursion_plugin_mod


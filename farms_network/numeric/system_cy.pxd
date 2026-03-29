cdef class ODESystemCy:

    cdef void evaluate(self, double time, double[:] states, double[:] derivatives) noexcept
    cdef void on_substep(self, double time, double h) noexcept


cdef class SDESystemCy:

    cdef void evaluate_a(self, double time, double[:] states, double[:] drift) noexcept
    cdef void evaluate_b(self, double time, double[:] states, double[:] diffusion) noexcept

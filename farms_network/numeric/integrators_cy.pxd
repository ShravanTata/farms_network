from farms_core.array.array_cy cimport DoubleArray1D
from libc.math cimport sqrt as csqrt

from .system_cy cimport ODESystem, SDESystem

include 'types.pxd'


cdef class Integrator:
    cdef:
        unsigned int dim
        double dt

    cdef void _step(self, ODESystem sys, double time, double[:] states) noexcept


cdef class EulerSolver(Integrator):
    cdef:
        DoubleArray1D derivatives


cdef class RK2Solver(Integrator):
    cdef:
        DoubleArray1D k1
        DoubleArray1D k2
        DoubleArray1D states_tmp


cdef class RK4Solver(Integrator):
    cdef:
        DoubleArray1D k1
        DoubleArray1D k2
        DoubleArray1D k3
        DoubleArray1D k4
        DoubleArray1D states_tmp


cdef class RK45Solver(Integrator):
    cdef:
        DoubleArray1D k1
        DoubleArray1D k2
        DoubleArray1D k3
        DoubleArray1D k4
        DoubleArray1D k5
        DoubleArray1D k6
        DoubleArray1D k7
        DoubleArray1D states_tmp


        double atol
        double rtol
        double max_step
        double h  # current sub-step size
        bint fsal_valid  # whether k1 is already computed (FSAL)

        # Diagnostics
        public unsigned long total_substeps
        public unsigned long total_rejections
        public unsigned long total_steps


cdef class EulerMaruyamaSolver:

    cdef:
        DoubleArray1D drift
        DoubleArray1D diffusion

        unsigned int dim
        double dt

    cdef:
        cdef void step(self, SDESystem sys, double time, double[:] state) noexcept

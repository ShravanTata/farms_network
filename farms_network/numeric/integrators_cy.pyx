import numpy as np

from ..core.options import IntegrationOptions

from libc.stdio cimport printf
from libc.math cimport fabs, pow as cpow


NPDTYPE = np.float64


cdef class Integrator:

    def __init__(self, unsigned int dim, double dt):
        self.dim = dim
        self.dt = dt

    cdef void _step(self, ODESystem sys, double time, double[:] states) noexcept:
        pass

    def step(self, ODESystem sys, double time, double[:] states):
        self._step(sys, time, states)


cdef class EulerSolver(Integrator):

    def __init__(self, unsigned int dim, double dt):
        super().__init__(dim, dt)
        self.derivatives = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )

    cdef void _step(self, ODESystem sys, double time, double[:] states) noexcept:
        cdef unsigned int i
        cdef double[:] derivatives = self.derivatives.array

        sys.evaluate(time, states, derivatives)
        for i in range(self.dim):
            states[i] = states[i] + self.dt * derivatives[i]


cdef class RK2Solver(Integrator):

    def __init__(self, unsigned int dim, double dt):
        super().__init__(dim, dt)
        self.k1 = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.k2 = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.states_tmp = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )

    cdef void _step(self, ODESystem sys, double time, double[:] states) noexcept:
        cdef unsigned int i
        cdef double dt = self.dt
        cdef double[:] k1 = self.k1.array
        cdef double[:] k2 = self.k2.array
        cdef double[:] states_tmp = self.states_tmp.array

        # Compute k1
        sys.evaluate(time, states, k1)

        # Compute k2 at midpoint
        for i in range(self.dim):
            states_tmp[i] = states[i] + (dt / 2.0) * k1[i]
        sys.evaluate(time + dt / 2.0, states_tmp, k2)

        # Update: midpoint method
        for i in range(self.dim):
            states[i] = states[i] + dt * k2[i]


cdef class RK4Solver(Integrator):

    def __init__ (self, unsigned int dim, double dt):
        super().__init__(dim, dt)
        self.k1 = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.k2 = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.k3 = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.k4 = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.states_tmp = DoubleArray1D(
            array=np.full(shape=dim, fill_value=0.0, dtype=NPDTYPE,)
        )

    cdef void _step(self, ODESystem sys, double time, double[:] states) noexcept:
        cdef unsigned int i
        cdef double dt2 = self.dt / 2.0
        cdef double dt6 = self.dt / 6.0
        cdef double[:] k1 = self.k1.array
        cdef double[:] k2 = self.k2.array
        cdef double[:] k3 = self.k3.array
        cdef double[:] k4 = self.k4.array
        cdef double[:] states_tmp = self.states_tmp.array

        # Compute k1
        sys.evaluate(time, states, k1)

        # Compute k2
        for i in range(self.dim):
            states_tmp[i] = states[i] + (dt2 * k1[i])
        sys.evaluate(time + dt2, states_tmp, k2)

        # Compute k3
        for i in range(self.dim):
            states_tmp[i] = states[i] + (dt2 * k2[i])
        sys.evaluate(time + dt2, states_tmp, k3)

        # Compute k4
        for i in range(self.dim):
            states_tmp[i] = states[i] + self.dt * k3[i]
        sys.evaluate(time + self.dt, states_tmp, k4)

        # Update y: y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        for i in range(self.dim):
            states[i] = states[i] + dt6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


cdef class RK45Solver(Integrator):
    """ Dormand-Prince RK45 adaptive integrator.
    Subdivides within a fixed dt window to meet error tolerances.
    Uses FSAL (First Same As Last) optimization. """

    # Dormand-Prince coefficients
    # a coefficients (time fractions)
    # a2=1/5, a3=3/10, a4=4/5, a5=8/9, a6=1, a7=1

    # b coefficients (stage weights) — see _step for inline usage

    # 5th order weights (for solution)
    # b1=35/384, b3=500/1113, b4=125/192, b5=-2187/6784, b6=11/84

    # Error coefficients (5th - 4th order)
    # e1=71/57600, e3=-71/16695, e4=71/1920, e5=-17253/339200, e6=22/525, e7=-1/40

    def __init__(self, unsigned int dim, double dt,
                 double atol=1e-4, double rtol=1e-4, double max_step=0.0):
        super().__init__(dim, dt)
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step if max_step > 0.0 else dt
        self.h = dt  # initial sub-step size
        self.fsal_valid = False
        self.total_substeps = 0
        self.total_rejections = 0
        self.total_steps = 0

        cdef array = np.zeros(dim, dtype=NPDTYPE)
        self.k1 = DoubleArray1D(array=array.copy())
        self.k2 = DoubleArray1D(array=array.copy())
        self.k3 = DoubleArray1D(array=array.copy())
        self.k4 = DoubleArray1D(array=array.copy())
        self.k5 = DoubleArray1D(array=array.copy())
        self.k6 = DoubleArray1D(array=array.copy())
        self.k7 = DoubleArray1D(array=array.copy())
        self.states_tmp = DoubleArray1D(array=array.copy())

    cdef void _step(self, ODESystem sys, double time, double[:] states) noexcept:
        """ Advance states by exactly self.dt using adaptive sub-stepping. """
        cdef unsigned int i
        cdef double t = time
        cdef double t_end = time + self.dt
        cdef double h = self.h
        cdef double err_norm, scale, h_new, tol
        cdef double[:] k1 = self.k1.array
        cdef double[:] k2 = self.k2.array
        cdef double[:] k3 = self.k3.array
        cdef double[:] k4 = self.k4.array
        cdef double[:] k5 = self.k5.array
        cdef double[:] k6 = self.k6.array
        cdef double[:] k7 = self.k7.array
        cdef double[:] st = self.states_tmp.array
        cdef unsigned int dim = self.dim

        # Clamp initial h to remaining interval
        if h > self.dt:
            h = self.dt

        cdef double abs_s, abs_st, err_i
        cdef unsigned long substeps = 0
        cdef unsigned long rejections = 0
        self.total_steps += 1

        while t < t_end:
            # Don't overshoot
            if t + h > t_end:
                h = t_end - t
            if h < 1e-15:
                break

            # Stage 1: FSAL — reuse k7 from previous accepted step
            if not self.fsal_valid:
                sys.evaluate(t, states, k1)
            # else k1 already contains the correct values

            # Stage 2
            for i in range(dim):
                st[i] = states[i] + h * (1.0/5.0) * k1[i]
            sys.evaluate(t + (1.0/5.0)*h, st, k2)

            # Stage 3
            for i in range(dim):
                st[i] = states[i] + h * ((3.0/40.0)*k1[i] + (9.0/40.0)*k2[i])
            sys.evaluate(t + (3.0/10.0)*h, st, k3)

            # Stage 4
            for i in range(dim):
                st[i] = states[i] + h * (
                    (44.0/45.0)*k1[i] - (56.0/15.0)*k2[i] + (32.0/9.0)*k3[i]
                )
            sys.evaluate(t + (4.0/5.0)*h, st, k4)

            # Stage 5
            for i in range(dim):
                st[i] = states[i] + h * (
                    (19372.0/6561.0)*k1[i] - (25360.0/2187.0)*k2[i]
                    + (64448.0/6561.0)*k3[i] - (212.0/729.0)*k4[i]
                )
            sys.evaluate(t + (8.0/9.0)*h, st, k5)

            # Stage 6
            for i in range(dim):
                st[i] = states[i] + h * (
                    (9017.0/3168.0)*k1[i] - (355.0/33.0)*k2[i]
                    + (46732.0/5247.0)*k3[i] + (49.0/176.0)*k4[i]
                    - (5103.0/18656.0)*k5[i]
                )
            sys.evaluate(t + h, st, k6)

            # 5th order solution (into st)
            for i in range(dim):
                st[i] = states[i] + h * (
                    (35.0/384.0)*k1[i] + (500.0/1113.0)*k3[i]
                    + (125.0/192.0)*k4[i] - (2187.0/6784.0)*k5[i]
                    + (11.0/84.0)*k6[i]
                )

            # Stage 7 (FSAL: evaluate at the new point)
            sys.evaluate(t + h, st, k7)

            # Error estimate (5th - 4th order)
            err_norm = 0.0
            for i in range(dim):
                err_i = h * (
                    (71.0/57600.0)*k1[i] - (71.0/16695.0)*k3[i]
                    + (71.0/1920.0)*k4[i] - (17253.0/339200.0)*k5[i]
                    + (22.0/525.0)*k6[i] - (1.0/40.0)*k7[i]
                )
                abs_s = fabs(states[i])
                abs_st = fabs(st[i])
                tol = self.atol + self.rtol * (abs_s if abs_s > abs_st else abs_st)
                err_norm += (err_i / tol) * (err_i / tol)
            err_norm = csqrt(err_norm / dim)

            substeps += 1

            if err_norm <= 1.0:
                # Accept step
                t += h
                for i in range(dim):
                    states[i] = st[i]


                # FSAL: k7 becomes k1 for the next step
                for i in range(dim):
                    k1[i] = k7[i]
                self.fsal_valid = True

                # Increase step size (with safety factor)
                if err_norm > 1e-15:
                    scale = 0.9 * cpow(1.0 / err_norm, 0.2)
                    if scale > 5.0:
                        scale = 5.0
                    h_new = h * scale
                else:
                    h_new = h * 5.0
                if h_new > self.max_step:
                    h_new = self.max_step
                h = h_new
            else:
                # Reject step — decrease step size and retry
                scale = 0.9 * cpow(1.0 / err_norm, 0.25)
                if scale < 0.2:
                    scale = 0.2
                h = h * scale
                self.fsal_valid = False
                rejections += 1

        # Save h and diagnostics
        self.h = h
        self.total_substeps += substeps
        self.total_rejections += rejections


cdef class EulerMaruyamaSolver:

    def __init__ (self, unsigned int dim, double dt):

        super().__init__()
        self.dim = dim
        self.dt = dt
        self.drift = DoubleArray1D(
            array=np.full(shape=self.dim, fill_value=0.0, dtype=NPDTYPE,)
        )
        self.diffusion = DoubleArray1D(
            array=np.full(shape=self.dim, fill_value=0.0, dtype=NPDTYPE,)
        )

    cdef void step(self, SDESystem sys, double time, double[:] state) noexcept:
        """ Update stochastic noise process with Euler–Maruyama method (also called the
        Euler method) is a method for the approximate numerical solution of a stochastic
        differential equation (SDE) """

        cdef unsigned int i
        cdef double[:] drift = self.drift.array
        cdef double[:] diffusion = self.diffusion.array

        sys.evaluate_a(time, state, drift)
        sys.evaluate_b(time, state, diffusion)
        for i in range(self.dim):
            state[i] += drift[i]*self.dt + csqrt(self.dt)*diffusion[i]

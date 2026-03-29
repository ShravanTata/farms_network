""" Python-level integrator interface.
Wraps Cython integrators for convenient top-level access without recompilation. """

from .integrators_cy import (EulerMaruyamaSolverCy, EulerSolverCy,
                             IntegratorCy, RK2SolverCy, RK4SolverCy,
                             RK45SolverCy)

# Re-export Cython classes directly — they are the implementations.
# The Python wrappers below add convenience without changing the interface.

Integrator: IntegratorCy = IntegratorCy
EulerSolver: EulerSolverCy = EulerSolverCy
RK2Solver: RK2SolverCy = RK2SolverCy
RK4Solver: RK4SolverCy = RK4SolverCy
EulerMaruyamaSolver: EulerMaruyamaSolverCy = EulerMaruyamaSolverCy


class RK45Solver(RK45SolverCy):
    """ Dormand-Prince RK45 adaptive integrator with Python-level diagnostics.

    Parameters
    ----------
    dim : int
        Number of state variables.
    dt : float
        Fixed outer timestep (the integrator subdivides internally).
    atol : float
        Absolute error tolerance.
    rtol : float
        Relative error tolerance.
    max_step : float
        Maximum internal sub-step size. Defaults to dt.
    """

    def __init__(self, dim, dt, atol=1e-4, rtol=1e-4, max_step=0.0):
        super().__init__(dim, dt, atol=atol, rtol=rtol, max_step=max_step)

    @property
    def diagnostics(self):
        """ Return integration diagnostics as a dict. """
        avg = self.total_substeps / self.total_steps if self.total_steps > 0 else 0.0
        return {
            "total_steps": self.total_steps,
            "total_substeps": self.total_substeps,
            "total_rejections": self.total_rejections,
            "avg_substeps_per_step": avg,
            "current_h": self.h,
        }

    def __repr__(self):
        return (
            f"RK45Solver(dim={self.dim}, dt={self.dt}, "
            f"atol={self.atol}, rtol={self.rtol}, max_step={self.max_step})"
        )


INTEGRATORS = {
    "euler": EulerSolver,
    "rk2": RK2Solver,
    "rk4": RK4Solver,
    "rk45": RK45Solver,
}


def from_options(integration_options, dim):
    """ Create an integrator from IntegrationOptions.

    Parameters
    ----------
    integration_options : IntegrationOptions
        Integration configuration.
    dim : int
        Number of state variables.

    Returns
    -------
    Integrator
    """
    name: str = integration_options.integrator
    dt: float = integration_options.timestep
    solver_cls: Integrator = INTEGRATORS.get(name)
    if solver_cls is None:
        raise ValueError(
            f"Unknown integrator '{name}'. Available: {list(INTEGRATORS.keys())}"
        )
    if name == "rk45":
        return solver_cls(
            dim, dt,
            atol=integration_options.atol,
            rtol=integration_options.rtol,
            max_step=integration_options.max_step if integration_options.max_step > 0 else dt,
        )
    return solver_cls(dim, dt)


__all__ = [
    "Integrator",
    "EulerSolver",
    "RK2Solver",
    "RK4Solver",
    "RK45Solver",
    "EulerMaruyamaSolver",
    "INTEGRATORS",
    "from_options",
]

Numerical Integrators
=====================

The ``farms_network.numeric.integrators`` module provides numerical integration schemes for advancing the neural network dynamics.

Available Integrators
---------------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 60

   * - Name
     - Order
     - Description
   * - ``euler``
     - 1
     - Forward Euler method. Cheapest per step (1 ODE evaluation). Suitable when timestep is small relative to dynamics.
   * - ``rk2``
     - 2
     - Midpoint method (2nd-order Runge-Kutta). 2 ODE evaluations per step.
   * - ``rk4``
     - 4
     - Classic 4th-order Runge-Kutta. 4 ODE evaluations per step. Good default for fixed-step integration.
   * - ``rk45``
     - 4/5
     - Dormand-Prince adaptive integrator. Subdivides within a fixed timestep window to meet error tolerances. Uses FSAL optimization (6 effective evaluations per accepted sub-step).

Configuration
-------------

Integrators are selected via ``IntegrationOptions``:

.. code-block:: python

   from farms_network.core.options import IntegrationOptions

   # Fixed-step RK4 (default)
   opts = IntegrationOptions.defaults(
       timestep=1e-3,
       integrator="rk4",
   )

   # Adaptive RK45
   opts = IntegrationOptions.defaults(
       timestep=1e-3,
       integrator="rk45",
       atol=1e-6,
       rtol=1e-4,
   )

The integrator is instantiated via ``Network.setup_integrator()``, which reads from ``IntegrationOptions``.

Integration Options
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``timestep``
     - 0.001
     - Fixed outer timestep (seconds)
   * - ``integrator``
     - ``"euler"``
     - Integration method: ``euler``, ``rk2``, ``rk4``, ``rk45``
   * - ``atol``
     - 1e-6
     - Absolute error tolerance (used by ``rk45``)
   * - ``rtol``
     - 1e-6
     - Relative error tolerance (used by ``rk45``)
   * - ``max_step``
     - 0.0
     - Maximum sub-step size for adaptive integrators (0 = use timestep)

RK45 Adaptive Integration
--------------------------

The ``rk45`` integrator advances by exactly ``timestep`` per ``step()`` call, but internally subdivides using adaptive sub-stepping to meet the specified ``atol`` and ``rtol``. This is designed for compatibility with external physics simulators that require fixed-interval stepping.

The integrator remembers its internal sub-step size between calls, so it adapts to the system's dynamics over time.

Diagnostics are available after integration:

.. code-block:: python

   network.setup_integrator()
   # ... run simulation ...
   print(network.solver.diagnostics)
   # {'total_steps': 10000, 'total_substeps': 10500,
   #  'total_rejections': 12, 'avg_substeps_per_step': 1.05,
   #  'current_h': 0.00098}

Noise Integration
-----------------

Stochastic noise (Ornstein-Uhlenbeck process) is integrated via Euler-Maruyama at each integration sub-step through the ``on_substep`` hook. This ensures the noise SDE is advanced with the correct step size, even with adaptive integrators.

For fixed-step integrators, noise is advanced by ``timestep`` once per step. For ``rk45``, noise is advanced at each accepted sub-step with the actual sub-step size.

API Reference
-------------

.. automodule:: farms_network.numeric.integrators
   :members:
   :undoc-members:

Network
=======

The ``Network`` class is the main entry point for building and simulating neural networks.

Basic Usage
-----------

.. code-block:: python

   from farms_network.core.network import Network
   from farms_network.core.options import NetworkOptions

   # Build from options
   network = Network.from_options(network_options)
   network.setup_integrator()

   # Simulation loop
   for iteration in range(n_iterations):
       time = iteration * timestep

       # Optionally update external inputs
       network.data.external_inputs.array[node_index] = value

       # Step and log
       network.step(time)
       network.update_logs(time)

   # Access results
   states = network.log.states.array      # (n_iterations, n_states)
   outputs = network.log.outputs.array    # (n_iterations, n_nodes)
   times = network.log.times.array        # (n_iterations,)

State Management
----------------

Reset
^^^^^

Reset states to initial values without reconstructing the network:

.. code-block:: python

   network.reset()
   network.setup_integrator()
   # Ready to run again with same topology

This zeros outputs, noise, and external inputs, restores states from the original options, and resets the iteration counter.

Rebuild
^^^^^^^

Modify the network topology and rebuild:

.. code-block:: python

   # Modify options
   network.options.add_node(new_node_options)
   network.options.add_edges(new_edge_options)

   # Rebuild from modified options
   network.rebuild()

   # Run again
   network.run()

``rebuild()`` re-creates all data arrays, C structures, and the integrator from the current ``network.options``. This is cheaper than constructing a new ``Network`` since it skips YAML parsing.

Coupling and Convergence Order
------------------------------

The network uses **explicit time-stepping** for inter-node coupling: each node's
``input_tf`` reads output values from the **previous** ``evaluate()`` call, not the
current one. This introduces an O(dt) coupling lag, making the overall system
**first-order accurate** for inter-node coupling regardless of integrator order.

This is standard practice in computational neuroscience simulators (Brian2, NEST).
Higher-order integrators (RK2, RK4) still improve accuracy for each node's *local*
dynamics (the ODE right-hand side), but the coupling between nodes remains first-order.

.. note::

   For higher coupling accuracy, reduce ``timestep``. The coupling error scales
   linearly with dt.

**When this matters:**

- Precise phase-locking studies requiring sub-degree accuracy
- Strong coupling with high-frequency dynamics
- Long simulations where small phase drift accumulates

**When this is fine (most use cases):**

- CPG locomotion circuits at typical timesteps (dt ≤ 1e-3)
- Qualitative behavior studies
- Networks where coupling strength is moderate

External Integrators
--------------------

When using an external integrator (e.g., scipy), call ``post_step()`` after each
step to advance noise and update logs:

.. code-block:: python

   from scipy.integrate import RK45

   ode_func = network.get_ode_func()
   for i in range(n_iterations):
       time = i * timestep
       # ... scipy integration ...
       network.post_step(time, timestep)

``post_step()`` handles noise advancement (Euler-Maruyama at the outer dt) and
log updates. Without it, noise states will not advance.

API Reference
-------------

.. autoclass:: farms_network.core.network.Network
   :members:
   :undoc-members:

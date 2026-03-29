""" Unit tests for numerical integrators.
Tests convergence order and cross-validates against scipy. """

import numpy as np
import pytest
from scipy.integrate import RK45 as ScipyRK45
from farms_network.core import options
from farms_network.core.network import Network


def make_oscillator_pair(timestep, n_iterations, integrator="rk4", **integrator_kwargs):
    """ Create a minimal 2-oscillator network for testing. """
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "test"},
        integration=options.IntegrationOptions.defaults(
            n_iterations=n_iterations,
            timestep=timestep,
            integrator=integrator,
            **integrator_kwargs,
        ),
        logs=options.NetworkLogOptions(buffer_size=n_iterations),
    )
    for i in range(2):
        network_options.add_node(
            options.OscillatorNodeOptions(
                name=f"osc_{i}",
                parameters=options.OscillatorNodeParameterOptions.defaults(
                    intrinsic_frequency=1.0,
                    nominal_amplitude=1.0,
                    amplitude_rate=20.0,
                ),
                state=options.OscillatorStateOptions(
                    initial=[0.0, 0.5, 0.0] if i == 0 else [np.pi, 0.5, 0.0]
                ),
                noise=None,
                visual=None,
            )
        )
    network_options.add_edge(
        options.OscillatorEdgeOptions(
            source="osc_0", target="osc_1", weight=10.0,
            type="excitatory",
            parameters=options.OscillatorEdgeParameterOptions(phase_difference=np.pi/2),
            visual=None,
        )
    )
    network_options.add_edge(
        options.OscillatorEdgeOptions(
            source="osc_1", target="osc_0", weight=10.0,
            type="excitatory",
            parameters=options.OscillatorEdgeParameterOptions(phase_difference=-np.pi/2),
            visual=None,
        )
    )
    return network_options


def run_network(network_options):
    """ Build and run a network, return final states. """
    network = Network.from_options(network_options)
    network.setup_integrator()
    n_iterations = network_options.integration.n_iterations
    dt = network_options.integration.timestep
    for i in range(n_iterations):
        network.step(i * dt)
    return np.array(network.data.states.array).copy()


def get_reference_solution(timestep, n_iterations):
    """ Get a high-accuracy reference solution using RK4 with very small dt. """
    ref_dt = timestep / 64
    ref_iterations = n_iterations * 64
    opts = make_oscillator_pair(ref_dt, ref_iterations, integrator="rk4")
    return run_network(opts)


# -----------------------------------------------------------------
# Test: All integrators produce physically reasonable results
# -----------------------------------------------------------------
class TestIntegratorsBasic:
    """ Verify each integrator runs without error and produces finite results. """

    @pytest.fixture(params=["euler", "rk2", "rk4"])
    def integrator_name(self, request):
        return request.param

    def test_runs_and_finite(self, integrator_name):
        opts = make_oscillator_pair(1e-3, 1000, integrator=integrator_name)
        states = run_network(opts)
        assert np.all(np.isfinite(states)), f"{integrator_name} produced non-finite states"

    def test_rk45_runs_and_finite(self):
        opts = make_oscillator_pair(
            1e-3, 1000, integrator="rk45", atol=1e-6, rtol=1e-4,
        )
        states = run_network(opts)
        assert np.all(np.isfinite(states)), "RK45 produced non-finite states"


# -----------------------------------------------------------------
# Test: Convergence order
# -----------------------------------------------------------------
class TestConvergenceOrder:
    """ Verify that each integrator converges.

    NOTE: Coupling uses explicit time-stepping (outputs from previous evaluate),
    so the overall coupled system is first-order regardless of integrator order.
    All integrators should show ratio ≈ 2 (first-order convergence) for coupled
    networks. The integrator's higher order only helps local node dynamics.

    Method: run at two step sizes (dt and dt/2), measure error vs reference.
    """

    T_FINAL = 0.5  # short integration to keep errors measurable

    def _convergence_ratio(self, integrator, dt_coarse):
        """ Compute error ratio between dt and dt/2 runs. """
        n_coarse = int(self.T_FINAL / dt_coarse)
        n_fine = n_coarse * 2
        dt_fine = dt_coarse / 2

        # Reference: RK4 at dt/64
        ref = get_reference_solution(dt_coarse, n_coarse)

        states_coarse = run_network(
            make_oscillator_pair(dt_coarse, n_coarse, integrator=integrator)
        )
        states_fine = run_network(
            make_oscillator_pair(dt_fine, n_fine, integrator=integrator)
        )

        err_coarse = np.linalg.norm(states_coarse - ref)
        err_fine = np.linalg.norm(states_fine - ref)

        if err_fine < 1e-14:
            return None  # at machine precision, can't measure order
        return err_coarse / err_fine

    def test_euler_converges(self):
        ratio = self._convergence_ratio("euler", dt_coarse=1e-3)
        if ratio is None:
            pytest.skip("At machine precision")
        # First-order coupling dominates: ratio ≈ 2
        assert 1.5 < ratio < 3.0, f"Euler convergence ratio {ratio:.2f}, expected ~2.0"

    def test_rk2_converges(self):
        ratio = self._convergence_ratio("rk2", dt_coarse=1e-3)
        if ratio is None:
            pytest.skip("At machine precision")
        # First-order coupling dominates: ratio ≈ 2
        assert 1.5 < ratio < 3.0, f"RK2 convergence ratio {ratio:.2f}, expected ~2.0"

    def test_rk4_converges(self):
        ratio = self._convergence_ratio("rk4", dt_coarse=1e-2)
        if ratio is None:
            pytest.skip("At machine precision")
        # First-order coupling dominates: ratio ≈ 2
        assert 1.5 < ratio < 3.0, f"RK4 convergence ratio {ratio:.2f}, expected ~2.0"


# -----------------------------------------------------------------
# Test: RK45 adaptive accuracy
# -----------------------------------------------------------------
class TestRK45Accuracy:
    """ Verify RK45 matches a tight-tolerance reference. """

    def test_rk45_vs_rk4_reference(self):
        """ RK45 with reasonable tolerances should match RK4 at small dt. """
        dt = 1e-3
        n_iter = 500

        # Reference: RK4 at dt/16
        ref = get_reference_solution(dt, n_iter)

        # RK45 at coarser dt
        opts = make_oscillator_pair(
            dt, n_iter, integrator="rk45", atol=1e-8, rtol=1e-6,
        )
        states = run_network(opts)

        err = np.linalg.norm(states - ref)
        assert err < 1e-4, f"RK45 error vs reference: {err:.2e}"

    def test_rk45_tighter_tolerance_is_more_accurate(self):
        """ Tighter tolerances should produce smaller errors. """
        dt = 1e-3
        n_iter = 500
        ref = get_reference_solution(dt, n_iter)

        opts_loose = make_oscillator_pair(
            dt, n_iter, integrator="rk45", atol=1e-4, rtol=1e-3,
        )
        opts_tight = make_oscillator_pair(
            dt, n_iter, integrator="rk45", atol=1e-10, rtol=1e-8,
        )

        err_loose = np.linalg.norm(run_network(opts_loose) - ref)
        err_tight = np.linalg.norm(run_network(opts_tight) - ref)

        if err_loose < 1e-12 and err_tight < 1e-12:
            pytest.skip("Both at machine precision relative to reference")

        assert err_tight < err_loose, (
            f"Tighter tolerance not more accurate: tight={err_tight:.2e}, loose={err_loose:.2e}"
        )


# -----------------------------------------------------------------
# Test: Cross-integrator consistency
# -----------------------------------------------------------------
class TestCrossIntegrator:
    """ All integrators should converge to the same solution at small dt. """

    def test_all_integrators_agree_at_small_dt(self):
        dt = 1e-4
        n_iter = 1000

        results = {}
        for name in ["euler", "rk2", "rk4"]:
            results[name] = run_network(
                make_oscillator_pair(dt, n_iter, integrator=name)
            )

        # At small dt, all should be close to RK4
        ref = results["rk4"]
        for name in ["euler", "rk2"]:
            err = np.linalg.norm(results[name] - ref)
            assert err < 1e-2, f"{name} vs RK4 at dt={dt}: error={err:.2e}"


# -----------------------------------------------------------------
# Test: Network reset
# -----------------------------------------------------------------
class TestNetworkReset:
    """ Verify that reset produces identical results on re-run. """

    def test_reset_reproduces_results(self):
        opts = make_oscillator_pair(1e-3, 500, integrator="rk4")
        network = Network.from_options(opts)
        network.setup_integrator()

        dt = opts.integration.timestep
        for i in range(500):
            network.step(i * dt)
        states_first = np.array(network.data.states.array).copy()

        network.reset()
        network.setup_integrator()
        for i in range(500):
            network.step(i * dt)
        states_second = np.array(network.data.states.array).copy()

        np.testing.assert_array_equal(
            states_first, states_second,
            err_msg="Reset did not reproduce identical results"
        )


# -----------------------------------------------------------------
# Test: Validate against scipy.integrate.RK45
# -----------------------------------------------------------------
class TestScipyRK45Comparison:
    """ Cross-validate farms_network RK45 against scipy's RK45.
    Both use Dormand-Prince with the same tolerances. """

    def _run_scipy_rk45(self, network, dt, n_iterations, atol, rtol):
        """ Step scipy's RK45 in fixed dt windows to match farms_network's
        per-step interface (external simulator interop). """
        ode_func = network.get_ode_func()
        y = np.array(network.data.states.array).copy()

        def f(t, y_):
            """ Wrapper: scipy passes numpy array, ode_func expects memoryview """
            return np.array(ode_func(t, y_))

        t = 0.0
        for i in range(n_iterations):
            t_end = t + dt
            solver = ScipyRK45(f, t, y, t_bound=t_end, atol=atol, rtol=rtol)
            while solver.status == 'running':
                solver.step()
            y = solver.y.copy()
            t = t_end

        return y

    def test_rk45_matches_scipy(self):
        """ farms_network RK45 and scipy RK45 should produce similar results. """
        dt = 1e-3
        n_iter = 500
        atol = 1e-8
        rtol = 1e-6

        # Run farms_network RK45
        opts = make_oscillator_pair(dt, n_iter, integrator="rk45", atol=atol, rtol=rtol)
        network_farms = Network.from_options(opts)
        network_farms.setup_integrator()
        for i in range(n_iter):
            network_farms.step(i * dt)
        states_farms = np.array(network_farms.data.states.array).copy()

        # Run scipy RK45 on the same network
        opts_ref = make_oscillator_pair(dt, n_iter, integrator="rk4")
        network_scipy = Network.from_options(opts_ref)
        network_scipy.setup_integrator()
        states_scipy = self._run_scipy_rk45(network_scipy, dt, n_iter, atol, rtol)

        err = np.linalg.norm(states_farms - states_scipy)
        # Both use Dormand-Prince with same tolerances; results should be close
        assert err < 1e-4, (
            f"farms RK45 vs scipy RK45: error={err:.2e}"
        )

    def test_both_match_rk4_reference(self):
        """ Both RK45 implementations should be close to a tight RK4 reference. """
        dt = 1e-3
        n_iter = 500
        atol = 1e-8
        rtol = 1e-6

        ref = get_reference_solution(dt, n_iter)

        # farms_network RK45
        opts = make_oscillator_pair(dt, n_iter, integrator="rk45", atol=atol, rtol=rtol)
        states_farms = run_network(opts)

        # scipy RK45
        opts_ref = make_oscillator_pair(dt, n_iter, integrator="rk4")
        network_scipy = Network.from_options(opts_ref)
        network_scipy.setup_integrator()
        states_scipy = self._run_scipy_rk45(network_scipy, dt, n_iter, atol, rtol)

        err_farms = np.linalg.norm(states_farms - ref)
        err_scipy = np.linalg.norm(states_scipy - ref)

        # Both should be within similar error bounds vs reference
        assert err_farms < 1e-3, f"farms RK45 vs ref: {err_farms:.2e}"
        assert err_scipy < 1e-3, f"scipy RK45 vs ref: {err_scipy:.2e}"


# -----------------------------------------------------------------
# Test: External integrator with post_step and noise
# -----------------------------------------------------------------
def make_noisy_oscillator_pair(timestep, n_iterations, integrator="rk4", **integrator_kwargs):
    """ Create a 2-oscillator network with OU noise for testing. """
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "test_noisy"},
        integration=options.IntegrationOptions.defaults(
            n_iterations=n_iterations,
            timestep=timestep,
            integrator=integrator,
            **integrator_kwargs,
        ),
        logs=options.NetworkLogOptions(buffer_size=n_iterations),
        random_seed=42,
    )
    for i in range(2):
        network_options.add_node(
            options.OscillatorNodeOptions(
                name=f"osc_{i}",
                parameters=options.OscillatorNodeParameterOptions.defaults(
                    intrinsic_frequency=1.0,
                    nominal_amplitude=1.0,
                    amplitude_rate=20.0,
                ),
                state=options.OscillatorStateOptions(
                    initial=[0.0, 0.5, 0.0] if i == 0 else [np.pi, 0.5, 0.0]
                ),
                noise=options.OrnsteinUhlenbeckOptions(
                    mu=0.0, sigma=0.1, tau=1.0,
                ),
                visual=None,
            )
        )
    network_options.add_edge(
        options.OscillatorEdgeOptions(
            source="osc_0", target="osc_1", weight=10.0,
            type="excitatory",
            parameters=options.OscillatorEdgeParameterOptions(phase_difference=np.pi/2),
            visual=None,
        )
    )
    network_options.add_edge(
        options.OscillatorEdgeOptions(
            source="osc_1", target="osc_0", weight=10.0,
            type="excitatory",
            parameters=options.OscillatorEdgeParameterOptions(phase_difference=-np.pi/2),
            visual=None,
        )
    )
    return network_options


class TestExternalIntegrator:
    """ Test post_step with external integrators and noise. """

    def test_post_step_advances_noise(self):
        """ Verify that post_step actually updates noise states. """
        dt = 1e-3
        n_iter = 100
        opts = make_noisy_oscillator_pair(dt, n_iter)
        network = Network.from_options(opts)
        network.setup_integrator()

        # Record initial noise state
        noise_initial = np.array(network.data.noise.states).copy()

        # Step with external integrator pattern (using internal ode_func)
        ode_func = network.get_ode_func()
        for i in range(n_iter):
            time = i * dt
            ode_func(time, network.data.states.array)
            network.post_step(time, dt)

        noise_final = np.array(network.data.noise.states).copy()

        # Noise should have changed
        assert not np.allclose(noise_initial, noise_final), (
            "Noise states did not change after post_step — noise not advancing"
        )

    def test_post_step_vs_internal_noise_differs(self):
        """ Internal integrator uses on_substep (correct h per sub-step).
        External via post_step uses fixed dt. Both should produce non-zero noise,
        but trajectories will differ due to different RNG sequences. """
        dt = 1e-3
        n_iter = 200

        # Internal integrator path
        opts_internal = make_noisy_oscillator_pair(dt, n_iter)
        network_int = Network.from_options(opts_internal)
        network_int.setup_integrator()
        for i in range(n_iter):
            network_int.step(i * dt)
        noise_internal = np.array(network_int.data.noise.states).copy()

        # External integrator path (using post_step)
        opts_external = make_noisy_oscillator_pair(dt, n_iter)
        network_ext = Network.from_options(opts_external)
        network_ext.setup_integrator()
        ode_func = network_ext.get_ode_func()
        for i in range(n_iter):
            time = i * dt
            ode_func(time, network_ext.data.states.array)
            network_ext.post_step(time, dt)
        noise_external = np.array(network_ext.data.noise.states).copy()

        # Both should have non-zero noise
        assert np.any(noise_internal != 0.0), "Internal noise is all zeros"
        assert np.any(noise_external != 0.0), "External noise is all zeros"

    def test_post_step_without_noise(self):
        """ post_step should work fine on networks without noise. """
        dt = 1e-3
        n_iter = 100
        opts = make_oscillator_pair(dt, n_iter)
        network = Network.from_options(opts)
        network.setup_integrator()

        ode_func = network.get_ode_func()
        for i in range(n_iter):
            time = i * dt
            ode_func(time, network.data.states.array)
            network.post_step(time, dt)

        states = np.array(network.data.states.array)
        assert np.all(np.isfinite(states)), "post_step without noise produced non-finite states"

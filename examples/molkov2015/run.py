""" Reproduce gait transitions from Molkov et al. 2015
DOI: 10.1371/journal.pcbi.1004270

Phase oscillator simplification of the 4-center CPG model.
Sweeps the drive parameter alpha to show speed-dependent
left-right coordination transitions.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from farms_network.core import options
from farms_network.core.network import Network
from farms_network.core.options import EdgeVisualOptions, NodeVisualOptions
from tqdm import tqdm

# ---- Coupling parameters from Table 1 ----
# Each entry: (a0, a1, b, k0, k1)
# A(alpha) = a0 + a1*alpha
# k(alpha) = (k0 + k1*alpha) / k_cpl

COUPLING = {
    "left_right": {
        "a0": 24.0, "a1": -2.4,
        "b": 1.0,
        "k0": 0.8, "k1": 0.0,
    },
    "fore_hind": {
        "a0": -4.0/3.0, "a1": 2.0/3.0,
        "b": -1.0,
        "k0": 1.0/3.0, "k1": 0.0,
    },
    "diagonal": {
        "a0": 4.0/3.0, "a1": -2.0/3.0,
        "b": -1.0,
        "k0": 1.0, "k1": -0.31/14.0,
    },
}

# Limb names and their coupling relationships
LIMBS = ["LF", "RF", "LH", "RH"]

# Coupling topology: (source, target, type)
EDGES = [
    # Left-Right
    ("LF", "RF", "left_right"),
    ("RF", "LF", "left_right"),
    ("LH", "RH", "left_right"),
    ("RH", "LH", "left_right"),
    # Fore-Hind
    ("LF", "LH", "fore_hind"),
    ("LH", "LF", "fore_hind"),
    ("RF", "RH", "fore_hind"),
    ("RH", "RF", "fore_hind"),
    # Diagonal
    ("LF", "RH", "diagonal"),
    ("RH", "LF", "diagonal"),
    ("RF", "LH", "diagonal"),
    ("LH", "RF", "diagonal"),
]


def build_network(dt, n_iterations, k_cpl=1.0, intrinsic_frequency=1.0):
    """ Build the quadruped phase oscillator network. """

    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "molkov2015"},
        integration=options.IntegrationOptions.defaults(
            n_iterations=n_iterations,
            timestep=dt,
            integrator="rk4",
        ),
        logs=options.NetworkLogOptions(buffer_size=n_iterations),
    )

    # Add limb oscillators with perturbed initial phases.
    # Exact multiples of pi are fixed points for ALL alpha (sin(nπ)=0),
    # so small perturbations are needed to break symmetry.
    eps = 0.1
    initial_phases = {
        "LF": 0.0 + eps,
        "RF": np.pi - eps,
        "LH": np.pi + eps,
        "RH": 0.0 - eps,
    }

    for limb in LIMBS:
        network_options.add_node(
            options.MolkovOscillatorNodeOptions(
                name=limb,
                parameters=options.MolkovOscillatorNodeParameterOptions.defaults(
                    intrinsic_frequency=intrinsic_frequency,
                ),
                state=options.MolkovOscillatorStateOptions(
                    initial=[initial_phases[limb]],
                ),
                noise=None,
                visual=NodeVisualOptions(),
            )
        )

    # Add coupling edges
    for source, target, coupling_type in EDGES:
        params = COUPLING[coupling_type]
        network_options.add_edge(
            options.MolkovOscillatorEdgeOptions(
                source=source,
                target=target,
                weight=1.0,  # weight absorbed into k0/k1/k_cpl
                type="phase_coupling",
                parameters=options.MolkovOscillatorEdgeParameterOptions(
                    a0=params["a0"],
                    a1=params["a1"],
                    b=params["b"],
                    k0=params["k0"],
                    k1=params["k1"],
                    k_cpl=k_cpl,
                ),
                visual=EdgeVisualOptions(),
            )
        )

    return network_options


def run_at_alpha(network, dt, n_iterations, alpha):
    """ Run the network at a fixed alpha and return final phases. """

    # Set alpha as external input to all oscillators
    for j in range(4):
        network.data.external_inputs.array[j] = alpha

    for i in range(n_iterations):
        network.step(i * dt)

    # Return final phases
    return np.array(network.data.states.array).copy()


def compute_phase_difference(phi_a, phi_b):
    """ Compute normalized phase difference in [0, 1]. """
    diff = (phi_a - phi_b) % (2 * np.pi)
    return diff / (2 * np.pi)


def bifurcation_diagram(
    alpha_range=(0.0, 14.0),
    n_alpha=200,
    dt=1e-3,
    n_settle=5000,
    n_measure=2000,
    k_cpl=1.0,
    intrinsic_frequency=1.0,
):
    """ Sweep alpha and compute steady-state phase differences.

    At each alpha value, the network runs for n_settle steps (transient)
    then n_measure steps (measurement). Initial conditions for each alpha
    are taken from the final state of the previous alpha (continuation).
    """

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    n_total = n_settle + n_measure

    # Build network
    network_options = build_network(dt, n_total, k_cpl, intrinsic_frequency)
    network = Network.from_options(network_options)
    network.setup_integrator()

    # Get node indices
    node_names = [n.name for n in network.nodes]
    idx = {name: node_names.index(name) for name in LIMBS}

    # Storage for phase differences
    lr_fore = np.zeros(n_alpha)   # LF-RF
    lr_hind = np.zeros(n_alpha)   # LH-RH
    fore_hind_l = np.zeros(n_alpha)  # LF-LH
    fore_hind_r = np.zeros(n_alpha)  # RF-RH

    for n, alpha in enumerate(tqdm(alphas, desc="Sweeping alpha")):
        # Set alpha
        for j in range(4):
            network.data.external_inputs.array[j] = alpha

        # Run transient + measurement
        for i in range(n_total):
            network.step(i * dt)

        # Extract final phases from log
        states = np.array(network.data.states.array).copy()
        phi = {}
        for limb in LIMBS:
            state_idx = network.data.states.indices[idx[limb]]
            phi[limb] = states[state_idx]

        # Compute phase differences (normalized to [0, 1])
        lr_fore[n] = compute_phase_difference(phi["LF"], phi["RF"])
        lr_hind[n] = compute_phase_difference(phi["LH"], phi["RH"])
        fore_hind_l[n] = compute_phase_difference(phi["LF"], phi["LH"])
        fore_hind_r[n] = compute_phase_difference(phi["RF"], phi["RH"])

        # Continuation: keep current states as initial for next alpha
        # (reset iteration counter but preserve states)
        network._network_cy.iteration = 0

    return alphas, lr_fore, lr_hind, fore_hind_l, fore_hind_r


def plot_bifurcation(alphas, lr_fore, lr_hind, fore_hind_l, fore_hind_r):
    """ Plot phase differences vs alpha. """

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Left-Right phase differences
    ax = axes[0]
    ax.scatter(alphas, lr_fore, s=2, c='blue', label='LF-RF')
    ax.scatter(alphas, lr_hind, s=2, c='red', label='LH-RH')
    ax.set_ylabel('Phase difference (normalized)')
    ax.set_title('Left-Right coordination')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='alternation')
    ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5, label='synchrony')

    # Fore-Hind phase differences
    ax = axes[1]
    ax.scatter(alphas, fore_hind_l, s=2, c='green', label='LF-LH')
    ax.scatter(alphas, fore_hind_r, s=2, c='orange', label='RF-RH')
    ax.set_ylabel('Phase difference (normalized)')
    ax.set_xlabel(r'Drive parameter $\alpha$')
    ax.set_title('Fore-Hind coordination')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)

    plt.suptitle('Molkov et al. 2015 - Phase oscillator gait transitions')
    plt.tight_layout()
    plt.savefig('bifurcation_diagram.png', dpi=150)
    plt.show()


def plot_time_series(dt, n_iterations, alpha, k_cpl=1.0, intrinsic_frequency=1.0):
    """ Plot phase time series at a specific alpha value. """

    network_options = build_network(dt, n_iterations, k_cpl, intrinsic_frequency)
    network = Network.from_options(network_options)
    network.setup_integrator()

    # Set alpha
    for j in range(4):
        network.data.external_inputs.array[j] = alpha

    # Run
    for i in range(n_iterations):
        time = i * dt
        network.step(time)
        network.update_logs(time)

    # Plot
    times = np.array(network.log.times.array)
    states = np.array(network.log.states.array)

    node_names = [n.name for n in network.nodes]
    idx = {name: node_names.index(name) for name in LIMBS}

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Phase evolution
    ax = axes[0]
    for limb in LIMBS:
        state_idx = network.data.states.indices[idx[limb]]
        phase = states[:, state_idx]
        ax.plot(times, np.sin(phase), label=limb)
    ax.set_ylabel('sin(phase)')
    ax.set_title(f'Phase oscillator outputs (alpha={alpha})')
    ax.legend()

    # Phase differences
    ax = axes[1]
    lf_idx = network.data.states.indices[idx["LF"]]
    rf_idx = network.data.states.indices[idx["RF"]]
    lh_idx = network.data.states.indices[idx["LH"]]
    rh_idx = network.data.states.indices[idx["RH"]]

    lr_diff = compute_phase_difference(states[:, lf_idx], states[:, rf_idx])
    fh_diff = compute_phase_difference(states[:, lf_idx], states[:, lh_idx])

    ax.plot(times, lr_diff, label='LF-RF (left-right)')
    ax.plot(times, fh_diff, label='LF-LH (fore-hind)')
    ax.set_ylabel('Phase difference')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(f'time_series_alpha_{alpha}.png', dpi=150)
    plt.show()


def calculate_arc_rad(source_pos, target_pos, base_rad=-0.1):
    """Calculate arc3 radius for edge based on node positions."""
    dx = target_pos[0] - source_pos[0]
    dy = target_pos[1] - source_pos[1]

    # Set curvature to zero if nodes are aligned horizontally or vertically
    if dx == 0 or dy == 0:
        return 0.0

    # Decide on curvature based on position differences
    if abs(dx) > abs(dy):
        # Horizontal direction - positive rad for up, negative for down
        return -base_rad if dy >= 0 else base_rad
    else:
        # Vertical direction - positive rad for right, negative for left
        return base_rad if dx >= 0 else base_rad


def update_edge_visuals(network_options):
    """ Update edge options """

    nodes = network_options.nodes
    edges = network_options.edges
    for edge in edges:
        base_rad = calculate_arc_rad(
            nodes[nodes.index(edge.source)].visual.position,
            nodes[nodes.index(edge.target)].visual.position,
        )
        edge.visual.connectionstyle = f"arc3,rad={base_rad*0.0}"
    return network_options


def plot_network(network_options):
    """ Plot only network """

    graph = nx.node_link_graph(
        network_options,
        directed=True,
        multigraph=False,
        name="name",
        source="source",
        target="target",
    )

    node_positions = nx.forceatlas2_layout(graph)
    for index, node in enumerate(network_options.nodes):
        node.visual.position[:2] = node_positions[node.name]

    plt.figure()
    sparse_array = nx.to_scipy_sparse_array(graph)
    sns.heatmap(
        sparse_array.todense(), cbar=False, square=True,
        linewidths=0.5,
        annot=True
    )

    plt.figure()

    _ = nx.draw_networkx_nodes(
        graph,
        pos={
            node: data["visual"]["position"][:2] for node, data in graph.nodes.items()
        },
        node_color=[data["visual"]["color"] for node, data in graph.nodes.items()],
        alpha=0.25,
        edgecolors="k",
        linewidths=2.0,
        node_size=[300*data["visual"]["radius"] for node, data in graph.nodes.items()],
    )
    nx.draw_networkx_labels(
        graph,
        pos={
            node: data["visual"]["position"][:2] for node, data in graph.nodes.items()
        },
        labels={node: data["visual"]["label"] for node, data in graph.nodes.items()},
        font_size=11.0,
        font_weight="bold",
        font_family="sans-serif",
        alpha=1.0,
    )
    nx.draw_networkx_edges(
        graph,
        pos={
            node: data["visual"]["position"][:2]
            for node, data in graph.nodes.items()
        },
        edge_color=[
            [0.3, 1.0, 0.3] if data["type"] == "excitatory" else [0.7, 0.3, 0.3]
            for edge, data in graph.edges.items()
        ],
        width=1.0,
        arrowsize=10,
        style="-",
        arrows=True,
        min_source_margin=5,
        min_target_margin=5,
        connectionstyle=[
            data["visual"]["connectionstyle"]
            for edge, data in graph.edges.items()
        ],
    )
    plt.show()


def two_osc_bifurcation(
    alpha_range: tuple = (0.0, 14.0),
    n_alpha: int = 200,
    dt: float = 1e-3,
    n_settle: int = 5000,
    k_cpl: float = 1.0,
    intrinsic_frequency: float = 1.0,
) -> tuple:
    """ Bidirectional α sweep for 2 oscillators with LR coupling only.

    Reveals all three regimes of Eq 26 (Molkov 2015):
      Regime 1  A > 2B        : only alternation stable (φ=π)
      Regime 2  −2B < A < 2B  : bistability — hysteresis between the branches
      Regime 3  A < −2B       : only synchrony stable (φ=0)

    Runs two continuation sweeps:
      forward  — starts near alternation (Δφ≈π), α increasing
      backward — starts near synchrony  (Δφ≈0),  α decreasing
    """
    def _build(initial_phases: list) -> Network:
        lr = COUPLING["left_right"]
        net_opts = options.NetworkOptions(
            directed=True, multigraph=False,
            graph={"name": "eq26"},
            integration=options.IntegrationOptions.defaults(
                n_iterations=n_settle, timestep=dt, integrator="rk4",
            ),
            logs=options.NetworkLogOptions(buffer_size=n_settle),
        )
        for i, name in enumerate(["osc_0", "osc_1"]):
            net_opts.add_node(
                options.MolkovOscillatorNodeOptions(
                    name=name,
                    parameters=options.MolkovOscillatorNodeParameterOptions.defaults(
                        intrinsic_frequency=intrinsic_frequency,
                    ),
                    state=options.MolkovOscillatorStateOptions(
                        initial=[initial_phases[i]],
                    ),
                    noise=None,
                    visual=NodeVisualOptions(),
                )
            )
        for src, tgt in [("osc_0", "osc_1"), ("osc_1", "osc_0")]:
            net_opts.add_edge(
                options.MolkovOscillatorEdgeOptions(
                    source=src, target=tgt, weight=1.0, type="phase_coupling",
                    parameters=options.MolkovOscillatorEdgeParameterOptions(
                        a0=lr["a0"], a1=lr["a1"], b=lr["b"],
                        k0=lr["k0"], k1=lr["k1"], k_cpl=k_cpl,
                    ),
                    visual=EdgeVisualOptions(),
                )
            )
        net = Network.from_options(net_opts)
        net.setup_integrator()
        return net

    def _measure(net: Network) -> float:
        phi_0 = float(net.data.states.array[net.data.states.indices[0]])
        phi_1 = float(net.data.states.array[net.data.states.indices[1]])
        return compute_phase_difference(phi_0, phi_1)

    def _run(net: Network, alpha: float, delta: float = 0.0) -> None:
        """ Advance one alpha step.

        delta is a small directional perturbation applied to phi_1 before
        integration.  It must stay within the correct basin in the bistable
        region but be large enough to trigger the transition in the unstable
        regime within n_settle steps.

        Forward sweep  (delta > 0): phi_1 increases → Δφ = phi_0−phi_1 decreases
                                     from π toward 0, helping the system cross to
                                     synchrony once π becomes unstable.
        Backward sweep (delta < 0): phi_1 decreases → Δφ increases from 0 toward
                                     π, helping the system escape synchrony once 0
                                     becomes unstable.
        """
        net.data.external_inputs.array[:] = alpha
        net.data.states.array[net.data.states.indices[1]] += delta
        for i in range(n_settle):
            net.step(i * dt)
        net._network_cy.iteration = 0

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    fwd = np.zeros(n_alpha)
    bwd = np.zeros(n_alpha)

    # The perturbation magnitude must satisfy two constraints:
    #   delta << (π − φ*) everywhere in the bistable region  [stay in basin]
    #   delta large enough that n_settle steps reach the other attractor in
    #   regime 3 / regime 1.
    # With COUPLING["left_right"] defaults (a0=24, a1=−2.4, B=1):
    #   bistable region α ∈ [9.17, 10.83]; φ* ranges from 0 to π.
    #   At the narrowest point (centre of bistable region α≈10, φ*=π/2),
    #   distance to separator = |π − π/2| = π/2 ≈ 1.57 rad >> 0.01 rad. ✓
    #   In regime 3 at α=11 (eigenvalue λ≈0.26), time to escape from 0.01 rad:
    #   T = ln(π / (2·0.01)) / (2k·λ) ≈ 5.1 / (2·0.8·0.26) ≈ 12 s = 12 000 steps.
    #   → transition appears near α≈11.5 where λ is large enough for n_settle=5000.
    delta = 0.01

    # Forward: start near alternation (Δφ ≈ π), sweep α upward
    eps = 0.05
    net_fwd = _build([eps, np.pi - eps])
    for n, alpha in enumerate(tqdm(alphas, desc="Eq26 forward")):
        _run(net_fwd, alpha, delta=+delta)
        fwd[n] = _measure(net_fwd)

    # Backward: start near synchrony (Δφ ≈ 0), sweep α downward
    net_bwd = _build([eps, eps])
    for n, alpha in enumerate(tqdm(alphas[::-1], desc="Eq26 backward")):
        _run(net_bwd, alpha, delta=-delta)
        bwd[n_alpha - 1 - n] = _measure(net_bwd)

    return alphas, fwd, bwd


def plot_eq26_regimes(alphas: np.ndarray, fwd: np.ndarray, bwd: np.ndarray) -> None:
    """ Plot bidirectional bifurcation showing all three Eq 26 regimes.

    Top panel  : phase difference vs α (forward=red, backward=blue).
    Bottom panel: coupling coefficient A(α) with ±2B thresholds annotated.
    """
    lr = COUPLING["left_right"]
    b = lr["b"]
    A_vals = lr["a0"] + lr["a1"] * alphas

    # Regime boundary α values: A = ±2B
    alpha_2b = (lr["a0"] - 2 * b) / (-lr["a1"])   # A crosses +2B
    alpha_m2b = (lr["a0"] + 2 * b) / (-lr["a1"])  # A crosses −2B

    # Fold: both 0.0 and 1.0 represent synchrony (Δφ=0 and Δφ=2π differ only by 2π)
    fwd_plot = np.minimum(fwd, 1.0 - fwd)
    bwd_plot = np.minimum(bwd, 1.0 - bwd)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(alphas, fwd_plot, color='tab:red',  lw=1.5, label='forward  (from alternation)')
    ax.plot(alphas, bwd_plot, color='tab:blue', lw=1.5, label='backward (from synchrony)', linestyle='--')
    ax.axhline(y=0.5, color='gray', linestyle='--', lw=0.8, label='alternation Δφ=π')
    ax.axhline(y=0.0, color='gray', linestyle=':',  lw=0.8, label='synchrony  Δφ=0')
    ax.axvspan(alphas[0],  alpha_2b,  alpha=0.07, color='tab:red',    label='Regime 1: alt only')
    ax.axvspan(alpha_2b,  alpha_m2b, alpha=0.07, color='tab:purple', label='Regime 2: bistable')
    ax.axvspan(alpha_m2b, alphas[-1],alpha=0.07, color='tab:blue',   label='Regime 3: sync only')
    ax.axvline(x=alpha_2b,  color='tab:orange', lw=1.2, linestyle='--')
    ax.axvline(x=alpha_m2b, color='tab:purple', lw=1.2, linestyle='--')
    ax.set_ylabel('Phase difference (normalized)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('Eq 26 — G(φ) = A·sin φ − B·sin 2φ  [all three regimes]')

    ax = axes[1]
    ax.plot(alphas, A_vals, 'k-', lw=2, label='A(α)')
    ax.axhline(y=2 * b,  color='tab:orange', linestyle='--', lw=1.2, label=f'+2B = {2*b}')
    ax.axhline(y=-2 * b, color='tab:purple', linestyle='--', lw=1.2, label=f'−2B = {-2*b}')
    ax.axhline(y=0, color='gray', lw=0.6, linestyle=':')
    ax.set_xlabel(r'Drive parameter α')
    ax.set_ylabel('A(α)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('eq26_regimes.png', dpi=150)
    plt.show()


def test_two_oscillators():
    """ Minimal 2-oscillator test to verify coupling works.
    Should show transition from anti-phase to synchrony as alpha increases. """

    dt = 1e-3
    n_iter = 20000
    k_cpl = 1.0

    for alpha in [0.0, 5.0, 10.0, 12.0, 14.0]:
        # Two oscillators with left-right coupling
        net_opts = options.NetworkOptions(
            directed=True, multigraph=False,
            graph={"name": "test_2osc"},
            integration=options.IntegrationOptions.defaults(
                n_iterations=n_iter, timestep=dt, integrator="rk4",
            ),
            logs=options.NetworkLogOptions(buffer_size=n_iter),
        )
        for i, name in enumerate(["osc_0", "osc_1"]):
            net_opts.add_node(
                options.MolkovOscillatorNodeOptions(
                    name=name,
                    parameters=options.MolkovOscillatorNodeParameterOptions.defaults(
                        intrinsic_frequency=1.0,
                    ),
                    state=options.MolkovOscillatorStateOptions(
                        initial=[0.1 if i == 0 else np.pi - 0.1],
                    ),
                    noise=None, visual=NodeVisualOptions(),
                )
            )
        # Bidirectional left-right coupling
        lr = COUPLING["left_right"]
        for src, tgt in [("osc_0", "osc_1"), ("osc_1", "osc_0")]:
            net_opts.add_edge(
                options.MolkovOscillatorEdgeOptions(
                    source=src, target=tgt, weight=1.0, type="phase_coupling",
                    parameters=options.MolkovOscillatorEdgeParameterOptions(
                        a0=lr["a0"], a1=lr["a1"], b=lr["b"],
                        k0=lr["k0"], k1=lr["k1"], k_cpl=k_cpl,
                    ),
                    visual=EdgeVisualOptions(),
                )
            )

        network = Network.from_options(net_opts)
        network.setup_integrator()
        for j in range(2):
            network.data.external_inputs.array[j] = alpha

        for i in range(n_iter):
            network.step(i * dt)

        phi_0 = network.data.states.array[network.data.states.indices[0]]
        phi_1 = network.data.states.array[network.data.states.indices[1]]
        diff = compute_phase_difference(phi_0, phi_1)

        A = lr["a0"] + lr["a1"] * alpha
        k = (lr["k0"] + lr["k1"] * alpha) / k_cpl
        print(f"alpha={alpha:5.1f}  A={A:7.2f}  k={k:5.3f}  "
              f"phase_diff={diff:.4f}  (0.5=anti-phase, 0.0=sync)")


if __name__ == "__main__":

    # ---- Eq 26: all three regimes via bidirectional sweep ----
    print("=== Eq 26 regimes (2-oscillator, bidirectional sweep) ===")
    alphas_eq26, fwd, bwd = two_osc_bifurcation(
        alpha_range=(0.0, 14.0),
        n_alpha=200,
        dt=1e-3,
        n_settle=5000,
        k_cpl=1.0,
        intrinsic_frequency=1.0,
    )
    plot_eq26_regimes(alphas_eq26, fwd, bwd)

    # ---- Diagnostic: verify coupling with 2 oscillators ----
    print("=== 2-oscillator diagnostic ===")
    test_two_oscillators()

    # ---- Parameters ----
    dt = 1e-3
    k_cpl = 0.1
    freq = 5.0  # Hz

    network_options = build_network(dt, 1000, k_cpl, freq)
    plot_network(network_options)

    # ---- Time series at specific alpha values ----
    print("=== Time series at alpha=2 (low speed) ===")
    plot_time_series(dt, 10000, alpha=2.0, k_cpl=k_cpl, intrinsic_frequency=freq)

    print("=== Time series at alpha=7 (medium speed) ===")
    plot_time_series(dt, 10000, alpha=7.0, k_cpl=k_cpl, intrinsic_frequency=freq)

    print("=== Time series at alpha=12 (high speed) ===")
    plot_time_series(dt, 10000, alpha=12.0, k_cpl=k_cpl, intrinsic_frequency=freq)

    # ---- Bifurcation diagram ----
    print("=== Bifurcation diagram ===")
    alphas, lr_fore, lr_hind, fore_hind_l, fore_hind_r = bifurcation_diagram(
        alpha_range=(0.0, 14.0),
        n_alpha=200,
        dt=dt,
        n_settle=5000,
        n_measure=2000,
        k_cpl=k_cpl,
        intrinsic_frequency=freq,
    )
    plot_bifurcation(alphas, lr_fore, lr_hind, fore_hind_l, fore_hind_r)

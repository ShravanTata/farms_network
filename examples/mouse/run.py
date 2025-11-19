""" Generate and reproduce Zhang, Shevtsova, et al. eLife 2022;11:e73424. DOI:
https://doi.org/10.7554/eLife.73424 paper network """

import numpy.matlib as npml
import seaborn as sns
from farms_core.io.yaml import read_yaml
from farms_core.utils import profile
from farms_network.core import options
from farms_network.core.network import Network
from farms_network.numeric.integrators_cy import RK4Solver
from scipy.integrate import ode
from tqdm import tqdm

from components import *
from components import limb_circuit


def generate_network(n_iterations: int):
    """Generate network"""

    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "mouse"},
        integration=options.IntegrationOptions.defaults(
            n_iterations=n_iterations,
            timestep=1.0,
        ),
        logs=options.NetworkLogOptions(
            n_iterations=n_iterations,
        ),
    )

    ##############
    # MotorLayer #
    ##############
    # read muscle config file
    muscles_config = read_yaml(
        "/Users/tatarama/projects/work/research/neuromechanics/quadruped/mice/mouse-locomotion/data/config/muscles/quadruped_siggraph.yaml"
    )

    def update_muscle_name(name: str) -> str:
        """Update muscle name format"""
        return name.replace("_", "-")

    muscles = {
        "left": {
            "hind": {"agonist": [], "antagonist": []},
            "fore": {"agonist": [], "antagonist": []},
        },
        "right": {
            "hind": {"agonist": [], "antagonist": []},
            "fore": {"agonist": [], "antagonist": []},
        },
    }

    for name, muscle in muscles_config["muscles"].items():
        side = muscle["side"]
        limb = muscle["limb"]
        function = muscle.get("function", "agonist")
        muscles[side][limb][function].append(
            {
                "name": join_str(name.split("_")[2:]),
                "type": muscle["type"],
                "abbrev": muscle["abbrev"],
            }
        )

    ###################################
    # Connect patterns and motorlayer #
    ###################################
    hind_muscle_patterns = {
        "bfa": ["EA", "EB"],
        "ip": ["FA", "FB"],
        "bfpst": ["FA", "EA", "FB", "EB"],
        "rf": ["EA", "FB", "EB"],
        "va": ["EA", "FB", "EB"],
        "mg": ["FA", "EA", "EB"],
        "sol": ["EA", "EB"],
        "ta": ["FA", "FB"],
        "ab": ["FA", "EA", "FB", "EB"],
        "gm_dorsal": ["FA", "EA", "FB", "EB"],
        "edl": ["FA", "EA", "FB", "EB"],
        "fdl": ["FA", "EA", "FB", "EB"],
    }

    fore_muscle_patterns = {
        "spd": ["FA", "EA", "FB", "EB"],
        "ssp": ["FA", "EA", "FB", "EB"],
        "abd": ["FA", "EA", "FB", "EB"],
        "add": ["FA", "EA", "FB", "EB"],
        "tbl": ["FA", "EA", "FB", "EB"],
        "tbo": ["FA", "EA", "FB", "EB"],
        "bbs": ["FA", "FB"],
        "bra": ["FA", "EA", "FB", "EB"],
        "ecu": ["FA", "EA", "FB", "EB"],
        "fcu": ["FA", "EA", "FB", "EB"],
    }

    # Generate rhythm centers
    scale = 1.0
    for side in ("left", "right"):
        for limb in ("fore", "hind"):
            # Rhythm
            rg_x, rg_y = 10.0, 7.5
            off_x = -rg_x if side == "left" else rg_x
            off_y = rg_y if limb == "fore" else -rg_y
            mirror_x = limb == "hind"
            mirror_y = side == "right"
            rhythm = RhythmGenerator(
                name=join_str((side, limb)),
                transform_mat=get_transform_mat(
                    angle=0,
                    off_x=off_x,
                    off_y=off_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((rhythm.nodes()).values())
            network_options.add_edges((rhythm.edges()).values())
            # Commissural
            comm_x, comm_y = rg_x - 7.0, rg_y + 0.0
            off_x = -comm_x if side == "left" else comm_x
            off_y = comm_y if limb == "fore" else -comm_y
            mirror_x = limb == "hind"
            mirror_y = side == "right"
            commissural = Commissural(
                name=join_str((side, limb)),
                transform_mat=get_transform_mat(
                    angle=0,
                    off_x=off_x,
                    off_y=off_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((commissural.nodes()).values())
            # Drive
            commissural_drive = CommissuralDrive(
                name=join_str((side, limb)),
                transform_mat=get_transform_mat(
                    angle=0,
                    off_x=off_x,
                    off_y=off_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((commissural_drive.nodes()).values())
            # Pattern
            pf_x, pf_y = rg_x + 0.0, rg_y + 7.5
            off_x = -pf_x if side == "left" else pf_x
            off_y = pf_y if limb == "fore" else -pf_y
            mirror_x = limb == "hind"
            mirror_y = side == "right"
            pattern = PatternFormation(
                name=join_str((side, limb)),
                transform_mat=get_transform_mat(
                    angle=0,
                    off_x=off_x,
                    off_y=off_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((pattern.nodes()).values())
            network_options.add_edges((pattern.edges()).values())

            rhythm_pattern_edges = connect_rhythm_pattern(base_name=join_str((side, limb)))
            network_options.add_edges(rhythm_pattern_edges.values())

            # Motor Layer
            motor_x = pf_x + 0.5 * max(
                len(muscles["left"][limb]["agonist"]),
                len(muscles["left"][limb]["antagonist"]),
            )
            motor_y = pf_y + 5.0

            # Determine the mirror_x and mirror_y flags based on side and limb
            mirror_x = True if limb == "hind" else False
            mirror_y = True if side == "right" else False

            # Create MotorLayer for each side and limb
            motor = MotorLayer(
                muscles=muscles[side][limb],
                name=join_str((side, limb)),
                transform_mat=get_transform_mat(
                    angle=0.0,
                    off_x=motor_x if side == "right" else -motor_x,
                    off_y=motor_y if limb == "fore" else -motor_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                ),
            )
            network_options.add_nodes((motor.nodes()).values())
            network_options.add_edges((motor.edges()).values())
        # LPSN
        lpsn_x = rg_x - 9.0
        lpsn_y = rg_y - 5.5
        off_x = -lpsn_x if side == "left" else lpsn_x
        off_y = lpsn_y
        mirror_y = side == "right"
        lpsn = LPSN(
            name=side,
            transform_mat=get_transform_mat(
                angle=0,
                off_x=off_x,
                off_y=off_y,
                mirror_y=mirror_y,
            ),
        )
        network_options.add_nodes((lpsn.nodes()).values())
        lpsn_drive = LPSNDrive(
            name=side,
            transform_mat=get_transform_mat(
                angle=0,
                off_x=off_x,
                off_y=off_y,
                mirror_x=mirror_x,
                mirror_y=mirror_y,
            ),
        )
        network_options.add_nodes((lpsn_drive.nodes()).values())

        # Connect pattern layer to motor layer
        for muscle, patterns in hind_muscle_patterns.items():
            pattern_edges = connect_pattern_motor_layer(
                base_name=join_str((side, "hind")), muscle=muscle, patterns=patterns
            )
            network_options.add_edges(pattern_edges.values())
        for muscle, patterns in fore_muscle_patterns.items():
            pattern_edges = connect_pattern_motor_layer(
                base_name=join_str((side, "fore")), muscle=muscle, patterns=patterns
            )
            network_options.add_edges(pattern_edges.values())

    #################################
    # Connect rhythm to commissural #
    #################################
    rg_commissural_edges = connect_rg_commissural()
    network_options.add_edges(rg_commissural_edges.values())

    ##############################
    # Connect fore and hind lpsn #
    ##############################
    fore_hind_edges = connect_fore_hind_circuits()
    network_options.add_edges(fore_hind_edges.values())

    edge_specs = []

    for side in ("left", "right"):
        for limb in ("fore", "hind"):
            edge_specs.extend([
                ((side, limb, "RG", "F", "DR"), (side, limb, "RG", "F"), 1.0, "excitatory"),
                ((side, limb, "RG", "E", "DR"), (side, limb, "RG", "E"), 1.0, "excitatory"),
                ((side, limb, "V0V", "DR"), (side, limb, "V0V"), -1.0, "inhibitory"),
                ((side, limb, "V0D", "DR"), (side, limb, "V0D"), -1.0, "inhibitory"),
            ])

        # Add the diagonal V0D connection
        edge_specs.append(
            ((side, "V0D", "diag", "DR"), (side, "V0D", "diag"), -1.0, "inhibitory")
        )

    # Create the edges using create_edges
    edges = create_edges(
        edge_specs,
        base_name="",
        visual_options=options.EdgeVisualOptions()
    )
    network_options.add_edges(edges.values())

    return network_options


def generate_limb_circuit(n_iterations: int):
    """ Generate limb circuit """
    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "mouse"},
        integration=options.IntegrationOptions.defaults(
            n_iterations=n_iterations,
            timestep=1.0,
        ),
        logs=options.NetworkLogOptions(
            n_iterations=n_iterations,
        ),
    )

    ##############
    # MotorLayer #
    ##############
    # read muscle config file
    muscles_config = read_yaml(
        "/Users/tatarama/projects/work/research/neuromechanics/quadruped/mice/mouse-locomotion/data/config/muscles/quadruped_siggraph.yaml"
    )

    ###################################
    # Connect patterns and motorlayer #
    ###################################
    hind_muscle_patterns = {
        "bfa": ["EA", "EB"],
        "ip": ["FA", "FB"],
        "bfpst": ["FA", "EA", "FB", "EB"],
        "rf": ["EA", "FB", "EB"],
        "va": ["EA", "FB", "EB"],
        "mg": ["FA", "EA", "EB"],
        "sol": ["EA", "EB"],
        "ta": ["FA", "FB"],
        "ab": ["FA", "EA", "FB", "EB"],
        "gm_dorsal": ["FA", "EA", "FB", "EB"],
        "edl": ["FA", "EA", "FB", "EB"],
        "fdl": ["FA", "EA", "FB", "EB"],
    }

    fore_muscle_patterns = {
        "spd": ["FA", "EA", "FB", "EB"],
        "ssp": ["FA", "EA", "FB", "EB"],
        "abd": ["FA", "EA", "FB", "EB"],
        "add": ["FA", "EA", "FB", "EB"],
        "tbl": ["FA", "EA", "FB", "EB"],
        "tbo": ["FA", "EA", "FB", "EB"],
        "bbs": ["FA", "FB"],
        "bra": ["FA", "EA", "FB", "EB"],
        "ecu": ["FA", "EA", "FB", "EB"],
        "fcu": ["FA", "EA", "FB", "EB"],
    }

    def update_muscle_name(name: str) -> str:
        """Update muscle name format"""
        return name.replace("_", "-")

    muscles = {
        "left": {
            "hind": {"agonist": [], "antagonist": []},
            "fore": {"agonist": [], "antagonist": []},
        },
        "right": {
            "hind": {"agonist": [], "antagonist": []},
            "fore": {"agonist": [], "antagonist": []},
        },
    }

    for name, muscle in muscles_config["muscles"].items():
        side = muscle["side"]
        limb = muscle["limb"]
        function = muscle.get("function", "agonist")
        muscles[side][limb][function].append(
            {
                "name": join_str(name.split("_")[2:]),
                "type": muscle["type"],
                "abbrev": muscle["abbrev"],
            }
        )

    network_options = limb_circuit(
        network_options,
        side="right",
        limb="hind",
        muscles=muscles,
        contacts=("PHALANGE",),
        transform_mat=get_translation_matrix(off_x=-25.0, off_y=0.0)
    )

    return network_options


def generate_quadruped_circuit(
        n_iterations: int
):
    # Main network
    network_options = options.NetworkOptions(
        directed=True,
        multigraph=False,
        graph={"name": "quadruped"},
        integration=options.IntegrationOptions.defaults(
            n_iterations=int(n_iterations),
            timestep=1.0,
        ),
        logs=options.NetworkLogOptions(
            buffer_size=int(n_iterations),
        ),
    )
    network_options = quadruped_circuit(network_options)
    return network_options


def run_network(*args):
    network_options = args[0]

    network = Network.from_options(network_options)
    iterations = network_options.integration.n_iterations
    timestep = network_options.integration.timestep
    network.setup_integrator()

    integrator = ode(network.get_ode_func()).set_integrator(
        u'dopri5',
        method=u'adams',
        max_step=0.0,
        # nsteps=0
    )
    nnodes = len(network_options.nodes)
    integrator.set_initial_value(np.zeros(len(network.data.states.array[:]),), 0.0)

    # print("Data ------------", np.array(network.network.data.states.array))

    # data.to_file("/tmp/sim.hdf5")

    # # Integrate
    states = np.ones((iterations, len(network.data.states.array[:])))*1.0
    states_tmp = np.zeros((len(network.data.states.array[:],)))
    outputs = np.ones((iterations, len(network.data.outputs.array[:])))*1.0
    # states[0, 2] = -1.0

    # for index, node in enumerate(network_options.nodes):
    #     print(index, node.name)
    # network.data.external_inputs.array[:] = np.ones((1,))*(iteration/iterations)*1.0
    drive_input_indices = [
        index
        for index, node in enumerate(network_options.nodes)
        if "BS_input" in node.name and node.model == "relay"
    ]
    inputs = np.zeros(np.shape(network.data.external_inputs.array[:]))

    # Network drive : Alpha
    time_vec = np.arange(0, iterations)*timestep
    drive = 1.0
    drive_vec = np.hstack(
        (np.linspace(0, 1.05, len(time_vec[::2])),
         np.linspace(1.05, 0, len(time_vec[::2])))
    )

    for iteration in tqdm(range(0, iterations), colour='green', ascii=' >='):
        time = iteration
        # network.step(network.ode, iteration*1e-3, network.data.states.array)
        # network.step()
        # states[iteration+1, :] = network.data.states.array
        # network.step()
        # network.evaluate(iteration*1e-3, states[iteration, :])

        network.data.nodes['BS_input'].external_input.values = drive_vec[iteration]*drive

        network.step(time)
        network.update_logs(time)
        # integrator.set_initial_value(integrator.y, integrator.t)
        # integrator.integrate(integrator.t+1.0)
        # network.data.states.array[:] = integrator.y
        # outputs[iteration, :] = network.data.outputs.array
        # states[iteration, :] = integrator.y# network.data.states.array
        # network._network_cy.update_iteration()


    # # Integrate
    # N_ITERATIONS = network_options.integration.n_iterations
    # # states = np.ones((len(network.data.states.array),)) * 1.0

    # # network_gui = NetworkGUI(data=data)
    # # network_gui.run()

    # inputs_view = network.data.external_inputs.array
    # drive_input_indices = [
    #     index
    #     for index, node in enumerate(network_options.nodes)
    #     if "DR" in node.name and node.model == "linear"
    # ]
    # inputs = np.zeros((len(inputs_view),))
    # for iteration in tqdm(range(0, N_ITERATIONS), colour="green", ascii=" >="):
    #     inputs[drive_input_indices] = 0.02
    #     inputs_view[:] = inputs
    #     # states = rk4(iteration * 1e-3, states, network.ode, step_size=1)
    #     # states = network.integrator.step(network, iteration * 1e-3, states)
    #     network.step()
    #     # states = network.ode(iteration*1e-3, states)
    #     # print(np.array(states)[0], network.data.states.array[0], network.data.derivatives.array[0])
    #     network.data.times.array[iteration] = iteration*1e-3
    #     # network.logging(iteration)

    # network.data.to_file("/tmp/network.h5")
    network_options.save("/tmp/network_options.yaml")

    return network


def plot_network(network_options):
    """ Plot only network """

    network_options = update_edge_visuals(network_options)
    graph = nx.node_link_graph(
        network_options,
        directed=True,
        multigraph=False,
        link="edges",
        name="name",
        source="source",
        target="target",
    )

    # plt.figure()
    # sparse_array = nx.to_scipy_sparse_array(graph)
    # sns.heatmap(
    #     sparse_array.todense()[50:75, 50:75], cbar=False, square=True,
    #     linewidths=0.5,
    #     annot=True
    # )
    plt.figure()
    pos_circular = nx.circular_layout(graph)
    pos_spring = nx.spring_layout(graph)
    pos_graphviz = nx.nx_agraph.pygraphviz_layout(graph)

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


def get_gait_plot_from_neuron_act(act):
    """ Get start and end times of neurons for gait plot. """
    act = np.reshape(act, (np.shape(act)[0], 1))
    act_binary = (np.array(act) > 0.1).astype(int)
    act_binary = np.logical_not(act_binary).astype(int)
    act_binary[0] = 0
    gait_cycle = []
    start = (np.where(np.diff(act_binary[:, 0]) == 1.))[0]
    end = (np.where(np.diff(act_binary[:, 0]) == -1.))[0]
    for id, val in enumerate(start[:len(end)]):
        # HARD CODED TIME SCALING HERE!!
        gait_cycle.append((val*0.001, end[id]*0.001 - val*0.001))
    return gait_cycle


def calc_on_offsets(time_vec, out):
    os_=((np.diff((out>0.1).astype(np.int64),axis=0)==1).T)
    of_=((np.diff((out>0.1).astype(np.int64),axis=0)==-1).T)
    onsets=npml.repmat(time_vec[:-1],out.shape[1],1)[os_]
    offsets=npml.repmat(time_vec[:-1],out.shape[1],1)[of_]
    leg_os=(npml.repmat(np.arange(out.shape[1]),len(time_vec)-1,1).T)[os_]
    leg_of=(npml.repmat(np.arange(out.shape[1]),len(time_vec)-1,1).T)[of_]

    times_os=np.stack((onsets,leg_os,np.arange(len(leg_os))),1)
    times_os=times_os[times_os[:,0].argsort()]
    times_of=np.stack((offsets,leg_of,np.arange(len(leg_of))),1)
    times_of=times_of[times_of[:,0].argsort()]

    times = np.concatenate((
                np.concatenate((times_os,np.ones((len(times_os),1))*0.0),1),
                np.concatenate((times_of,np.ones((len(times_of),1))*1.0),1)))
    times=times[times[:,0].argsort()]
    return times


def calc_phase(time_vec, out, phase_diffs):
    times = calc_on_offsets(time_vec,out)
    ref_onsets = times[np.logical_and(times[:,1]==0,times[:,3]==0)][:,0]
    phase_dur=np.append(ref_onsets[1:]-ref_onsets[:-1],np.nan)

    p = times[times[:,1]==0]
    indices = np.where(np.diff(p[:,3])==1)
    fl_phase_dur = np.zeros((len(ref_onsets)))
    fl_phase_dur[:] = np.nan
    fl_phase_dur[p[indices,2].astype(int)] = p[[ind+1 for ind in indices],0] - p[indices,0]
    ex_phase_dur = phase_dur-fl_phase_dur

    M = np.zeros((len(ref_onsets),out.shape[1]))
    M[:] = np.nan
    M[:,0]=ref_onsets


    for i in range(1,out.shape[1]):
        p = times[np.logical_and((times[:,1]==0) | (times[:,1]==i),times[:,3]==0)]
        indices = np.where(np.diff(p[:,1])==i)
        M[p[indices,2].astype(int),i] = p[[ind+1 for ind in indices],0]


    phases=np.zeros((len(ref_onsets),len(phase_diffs)))
    for i,(x,y) in enumerate(phase_diffs):
        phases[:,i] = ((M[:,y]-M[:,x])/phase_dur)  % 1.0

    if phases.shape[0]!=0:
        no_nan = ~np.isnan(np.concatenate(
                    (np.stack((phase_dur,fl_phase_dur,ex_phase_dur),1),phases),1
                    )).any(axis=1)
        return (phase_dur[no_nan],fl_phase_dur[no_nan],ex_phase_dur[no_nan],phases[no_nan],ref_onsets[no_nan])
    else:
        return (phase_dur,fl_phase_dur,ex_phase_dur,phases,ref_onsets[:-1])


def plot_analysis(network: Network, network_options):
    """ Plot analysis """
    plot_names = [
        'right_fore_RG_F',
        'left_fore_RG_F',
        'right_hind_RG_F',
        'left_hind_RG_F',
    ]

    plot_traces = [
        network.log.nodes[name].output.values for name in plot_names
    ]

    _split_ramp = int(len(network.log.times.array)/2)
    phases_up = calc_phase(
        network.log.times.array[:_split_ramp],
        (np.asarray(plot_traces[:4]).T)[:_split_ramp],
        ((3, 2), (1, 0), (3, 1), (3, 0))
    )
    phases_down = calc_phase(
        network.log.times.array[_split_ramp:],
        (np.asarray(plot_traces[:4]).T)[_split_ramp:],
        ((3, 2), (1, 0), (3, 1), (3, 0))
    )

    alpha_vec = np.array(network.log.nodes["BS_input"].output.values)

    fig, ax = plt.subplots(4, 1, sharex='all')
    for j in range(4):
        ax[j].plot(alpha_vec[np.int32(phases_up[4])], phases_up[3][:, j], 'b*')
        ax[j].plot(alpha_vec[np.int32(phases_down[4])], phases_down[3][:, j], 'r*')

    fig, ax = plt.subplots(len(plot_names)+2, 1, sharex='all')
    #fig.canvas.set_window_title('Model Performance')
    fig.suptitle('Model Performance', fontsize=12)
    time_vec = np.array(network.log.times.array)
    for i, tr in enumerate(plot_traces):
        ax[i].plot(time_vec*0.001, np.array(tr), 'b', linewidth=1)
        ax[i].grid('on', axis='x')
        ax[i].set_ylabel(plot_names[i], fontsize=10)
        ax[i].set_yticks([0, 1])

    _width = 0.2
    colors = ['blue', 'green', 'red', 'black']
    for i, tr in enumerate(plot_traces):
        if i > 3:
            break
        ax[len(plot_names)].broken_barh(get_gait_plot_from_neuron_act(tr),
                                        (1.6-i*0.2, _width), facecolors=colors[i])

    ax[len(plot_names)].broken_barh(get_gait_plot_from_neuron_act(plot_traces[3]),
                                    (1.0, _width*4), facecolors=(0.2, 0.2, 0.2), alpha=0.5)
    ax[len(plot_names)].set_ylim(1.0, 1.8)
    ax[len(plot_names)].set_xlim(0)
    ax[len(plot_names)].set_xlabel('Time')
    ax[len(plot_names)].set_yticks([1.1, 1.3, 1.5, 1.7])
    ax[len(plot_names)].set_yticklabels(['RF', 'LF', 'RH', 'LH'])
    ax[len(plot_names)].grid(True)

    ax[len(plot_names)+1].fill_between(time_vec*0.001, 0, alpha_vec,
                                       color=(0.2, 0.2, 0.2), alpha=0.5)
    ax[len(plot_names)+1].grid('on', axis='x')
    ax[len(plot_names)+1].set_ylabel('ALPHA')
    ax[len(plot_names)+1].set_xlabel('Time [s]')

    plt.show()


def plot_data(network, network_options):
    plot_nodes = [
        index
        for index, node in enumerate(network.data.nodes)
        if ("RG_F" in node.name) and ("DR" not in node.name)
    ]

    plt.figure()

    for index, node_index in enumerate(plot_nodes):
        plt.fill_between(
            np.array(network.log.times.array)*1e-3,
            index + np.array(network.log.nodes[node_index].output.values),
            index,
            alpha=0.2,
            lw=1.0,
        )
        plt.plot(
            np.array(network.log.times.array)*1e-3,
            index + np.array(network.log.nodes[node_index].output.values),
            label=network.log.nodes[node_index].name,
        )
    plt.legend()

    plot_nodes = [
        index
        for index, node in enumerate(network.log.nodes)
        if ("Mn" in node.name)
    ]
    plt.figure()
    for index, node_index in enumerate(plot_nodes):
        plt.fill_between(
            np.array(network.log.times.array)*1e-3,
            index + np.array(network.log.nodes[node_index].output.values),
            index,
            alpha=0.2,
            lw=1.0,
        )
        plt.plot(
            np.array(network.log.times.array)*1e-3,
            index + np.array(network.log.nodes[node_index].output.values),
            label=network.log.nodes[node_index].name,
        )
    plt.legend()
    plt.show()


def main():
    """Main."""

    # Generate the network
    # network_options = generate_network(int(1e4))
    # network_options = generate_limb_circuit(int(5e4))
    network_options = generate_quadruped_circuit((1e3))

    # plot_network(network_options)
    network = run_network(network_options)
    plot_data(network, network_options)
    plot_analysis(network, network_options)


    # from abstract_control.control.generate import quadruped_siggraph_network
    # from copy import deepcopy
    # og_graph = quadruped_siggraph_network()

    # def update_names(old_names):
    #     replace_names = {
    #         "IIIn": "II_In",
    #         "IbIn": "Ib_In",
    #         "IaIn": "Ia_In",
    #         "_motor": "",
    #     }
    #     new_names = {}
    #     for name in old_names:
    #         new_name = deepcopy(name)
    #         for old, new in replace_names.items():
    #             new_name = new_name.replace(old, new)
    #         new_names[name] = new_name
    #     return new_names

    # new_names = update_names(og_graph.nodes)
    # og_graph = nx.relabel_nodes(og_graph, mapping=new_names)

    # print(f" OG edges {len(og_graph.edges)}")
    # print(f" new edges {len(graph.edges)}")

    # check_edges = 0
    # for edge in graph.edges():
    #     if edge in og_graph.edges:
    #         pass
    #     else:
    #         check_edges += 1
    #         print(f"{edge} not found...")
    # print(f"Check edges {check_edges}")


if __name__ == "__main__":
    profile.profile(main, profile_filename="/tmp/network.prof")
    # main()

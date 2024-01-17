import os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

def compute_bold_signal(spike_times, spike_indices, num_neurons, simulation_duration, TR):
    # Hemodynamic model parameters
    k1 = 10
    k2 = 3
    k3 = 3
    tau = 2.0  # Time constant in seconds
    alpha = 0.32
    E0 = 0.34

    # Convert spike times to BOLD signal
    bold_signal = np.zeros((num_neurons, int(simulation_duration/TR)))
    for neuron in range(num_neurons):
        spikes = np.zeros(int(simulation_duration/TR))
        spike_indices_for_neuron = spike_times[spike_indices == neuron]
        for spike_time in spike_indices_for_neuron:
            spikes[int(spike_time/TR)] = 1
        hrf = balloon_windkessel_hrf(TR, simulation_duration, alpha, E0, tau, k1, k2, k3)
        bold_signal[neuron, :] = np.convolve(spikes, hrf)[:int(simulation_duration/TR)]

    return bold_signal

def balloon_windkessel_hrf(TR, simulation_duration, alpha, E0, tau, k1, k2, k3):
    t = np.arange(0, float(simulation_duration), TR)
    hrf = np.zeros_like(t)

    # Double gamma HRF parameters
    peak_time = 6.0
    undershoot_time = 16.0
    peak_dispersion = 1.0
    undershoot_dispersion = 1.0
    undershoot_scale = 6.0

    for i in range(1, len(t)):
        peak = ((t[i] / peak_time) ** peak_dispersion) * np.exp(-(t[i] - peak_time) / peak_time)
        undershoot = ((t[i] / undershoot_time) ** undershoot_dispersion) * np.exp(-(t[i] - undershoot_time) / undershoot_time)
        hrf[i] = peak - (undershoot / undershoot_scale)

    return hrf

def create_input_matrix(num_neurons, simulation_duration, TR, stimulus_start, stimulus_duration, stimulus_value):
    num_time_points = int(simulation_duration / TR)
    input_matrix = np.zeros((num_neurons, num_time_points))
    stimulus_start_idx = int(stimulus_start / TR)
    stimulus_end_idx = int((stimulus_start + stimulus_duration) / TR)
    input_matrix[:, stimulus_start_idx:stimulus_end_idx] = stimulus_value
    return input_matrix

num_neurons = 100
simulation_duration = 120 * second
baseline_duration = 500 * ms
stimulus_duration = 1 * second
post_stimulus_duration = 500 * ms

conn_types = ['A', 'B', 'C']

eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''

G = NeuronGroup(num_neurons, eqs, threshold='v>1', reset='v=0', method='linear')
G.v = 'rand()'
G.I = 0
G.tau = 10 * ms

synapses = {}

spike_times_A = []
spike_indices_A = []
spike_times_B = []
spike_indices_B = []
spike_times_C = []
spike_indices_C = []

connectivity_matrices = {}

for conn_type in conn_types:
    S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
    if conn_type == 'A':
        S.connect(condition='i != j and i < num_neurons and j < num_neurons')
    elif conn_type == 'B':
        S.connect(condition='i != j and i >= num_neurons and i < 2*num_neurons and j >= num_neurons and j < 2*num_neurons')
    elif conn_type == 'C':
        S.connect(condition='i != j and i >= 2*num_neurons and j >= 2*num_neurons')
    if conn_type == 'A':
        S.w = 'rand() * 0.2'
    elif conn_type == 'B':
        S.w = 'rand() * 0.3'
    elif conn_type == 'C':
        S.w = 'rand() * 0.1'

    connectivity_matrices[conn_type] = np.array(S.w)

    synapses[conn_type] = S

M = StateMonitor(G, 'v', record=True)
spikemon_A = SpikeMonitor(G[:num_neurons])
spikemon_B = SpikeMonitor(G[num_neurons:2 * num_neurons])
spikemon_C = SpikeMonitor(G[2 * num_neurons:])

# Create a network to hold all the components
net = Network(collect())
net.add(G, M, spikemon_A, spikemon_B, spikemon_C, *synapses.values())

net.store()  # Store the initial state of the network

net.run(baseline_duration)  # Baseline period
G.I[:num_neurons] = 2.5  # Turn on the stimulus for neural population A
net.run(stimulus_duration)  # Stimulus period
G.I[:num_neurons] = 0  # Turn off the stimulus for neural population A
spike_times_A = spikemon_A.t
spike_indices_A = spikemon_A.i
spikemon_A.active = False  # Deactivate the spike monitor
net.run(post_stimulus_duration)  # Post-stimulus period

net.run(baseline_duration)  # Baseline period
G.I[num_neurons:2*num_neurons] = 3.0  # Turn on the stimulus for neural population B
spikemon_B.active = True  # Reactivate the spike monitor
net.run(stimulus_duration)  # Stimulus period
G.I[num_neurons:2*num_neurons] = 0  # Turn off the stimulus for neural population B
spike_times_B = spikemon_B.t
spike_indices_B = spikemon_B.i
spikemon_B.active = False  # Deactivate the spike monitor
net.run(post_stimulus_duration)  # Post-stimulus period

net.run(baseline_duration)  # Baseline period
G.I[2 * num_neurons:] = 1.5  # Turn on the stimulus for neural population C
spikemon_C.active = True  # Reactivate the spike monitor
net.run(stimulus_duration)  # Stimulus period
G.I[2 * num_neurons:] = 0  # Turn off the stimulus for neural population C
spike_times_C = spikemon_C.t
spike_indices_C = spikemon_C.i
spikemon_C.active = False  # Deactivate the spike monitor
net.run(post_stimulus_duration)  # Post-stimulus period

# Compute the BOLD signal from spiking data for each connectivity type
TR = 0.1  # The repetition time in seconds for the fMRI
bold_A = compute_bold_signal(spike_times_A, spike_indices_A, num_neurons, simulation_duration, TR)
bold_B = compute_bold_signal(spike_times_B, spike_indices_B, num_neurons, simulation_duration, TR)
bold_C = compute_bold_signal(spike_times_C, spike_indices_C, num_neurons, simulation_duration, TR)

# Save the BOLD signals
np.save("C:/Users/louis/Desktop/Everything/Yale/2022-23/S&DS492/simdata/bold_A.npy", bold_A)
np.save("C:/Users/louis/Desktop/Everything/Yale/2022-23/S&DS492/simdata/bold_B.npy", bold_B)
np.save("C:/Users/louis/Desktop/Everything/Yale/2022-23/S&DS492/simdata/bold_C.npy", bold_C)

# Save the connectivity matrices
np.save("C:/Users/louis/Desktop/Everything/Yale/2022-23/S&DS492/simdata/connectivity_matrix_A.npy", connectivity_matrices['A'])
np.save("C:/Users/louis/Desktop/Everything/Yale/2022-23/S&DS492/simdata/connectivity_matrix_B.npy", connectivity_matrices['B'])
np.save("C:/Users/louis/Desktop/Everything/Yale/2022-23/S&DS492/simdata/connectivity_matrix_C.npy", connectivity_matrices['C'])

# Plot the BOLD signals
time_points = np.arange(0, float(simulation_duration), TR)

plt.figure(figsize=(12, 4))
plt.plot(time_points, bold_A.T)
plt.xlabel('Time (s)')
plt.ylabel('BOLD signal')
plt.title('BOLD signal for connectivity type A')
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(time_points, bold_B.T)
plt.xlabel('Time (s)')
plt.ylabel('BOLD signal')
plt.title('BOLD signal for connectivity type B')
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(time_points, bold_C.T)
plt.xlabel('Time (s)')
plt.ylabel('BOLD signal')
plt.title('BOLD signal for connectivity type C')
plt.show()

stimulus_start_A = baseline_duration
stimulus_start_B = baseline_duration * 2 + stimulus_duration + post_stimulus_duration
stimulus_start_C = baseline_duration * 3 + stimulus_duration * 2 + post_stimulus_duration * 2

stimulus_value_A = 2.5
stimulus_value_B = 3.0
stimulus_value_C = 1.5

# Create the input matrices for each neural population
u_A = create_input_matrix(num_neurons, simulation_duration, TR, stimulus_start_A, stimulus_duration, stimulus_value_A)
u_B = create_input_matrix(num_neurons, simulation_duration, TR, stimulus_start_B, stimulus_duration, stimulus_value_B)
u_C = create_input_matrix(num_neurons, simulation_duration, TR, stimulus_start_C, stimulus_duration, stimulus_value_C)

np.save('C:/Users/louis/Desktop/Everything/Yale/2022-23/S&DS492/simdata/U_A.npy', u_A)
np.save('C:/Users/louis/Desktop/Everything/Yale/2022-23/S&DS492/simdata/U_B.npy', u_B)
np.save('C:/Users/louis/Desktop/Everything/Yale/2022-23/S&DS492/simdata/U_C.npy', u_C)

# Extract membrane potential values for populations A, B, and C
v_A = M.v[:num_neurons, :]
v_B = M.v[num_neurons:2 * num_neurons, :]
v_C = M.v[2 * num_neurons:, :]

# Concatenate membrane potential values along the second axis to form the x matrix
x_matrix = np.concatenate((v_A, v_B, v_C), axis=1)

np.save('C:/Users/louis/Desktop/Everything/Yale/2022-23/S&DS492/simdata/X_MATRIX.npy', x_matrix)
from datetime import datetime
from os import mkdir
import logging
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import nengo
import numpy as np
import pandas as pd
from tqdm import tqdm

exec(open("conf.py").read())

def fmt_num(num, width=18):
    """
    Format number to string.
    """

    return str(num)[:width].ljust(width)


def inputs_function(x):
    return x * tau_wm


def noise_decision_function(t):
    return np.random.normal(0.0, noise_decision)


def noise_bias_function(t):
    return np.random.normal(0.0, noise_wm)


def time_function(t):
    return time_scale if t > t_cue else 0


def decision_function(x):
    output = 0.0
    value = x[0] + x[1]
    if value > 0.0:
        output = 1.0
    elif value < 0.0:
        output = -1.0
    return output
    #return 1.0 if x[0] + x[1] > 0.0 else -1.0


class Alpha():
    """
    Base class for alpha receptors. Not to be used directly.
    """

    def __init__(self):
        self.x = steps
        self.y = 1 / (1 + (999 * np.exp(-0.1233 * (self.x / self.offset))))

        self.gains = []
        self.biass = []

        for i in range(len(steps)):
            self.gains.append(1 + self.gaind * self.y[i])
            self.biass.append(1 + self.biasd * self.y[i])

    def plot(self):
        out = f"./out/{self.__class__.__name__}"

        title = "Norepinepherine Concentration vs Neuron Activity in " + \
            self.pretty
        logging.info("Plotting " + title)
        plt.figure()
        plt.plot(self.x, self.y)

        plt.xlabel("Norepinephrine concentration (nM)")
        plt.ylabel("Activity (%)")
        plt.title(title)

        plt.vlines(self.ki, 0, 1, linestyles="dashed")
        plt.text(1.1 * self.ki, 0.1, "Affinity")

        plt.hlines(0.5, 0, 1000, linestyles="dashed")
        plt.text(1, 0.51, "50%")

        plt.xscale("log")
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.draw()
        plt.savefig(f"{out}-norep-activity.png")

        #######################################################################

        title = "Concentration vs Gain Scalar in" + self.pretty
        logging.info("Plotting " + title)
        plt.figure()
        plt.plot(self.x, self.gains)

        plt.xlabel("Norepinephrine concentration (nM)")
        plt.ylabel("Gain")
        plt.title(title)

        plt.xscale("log")

        plt.draw()
        plt.savefig(f"{out}-concentration-gain.png")

        #######################################################################

        title = "Concentration vs Bias scalar in " + self.pretty
        logging.info("Plotting " + title)
        plt.figure()
        plt.plot(self.x, self.biass)

        plt.xscale("log")

        plt.xlabel("Norepinephrine concentration (nM)")
        plt.ylabel("Bias")
        plt.title(title)

        plt.draw()
        plt.savefig(f"{out}-concentration-bias.png")


class Alpha1(Alpha):
    """
    Subclass of Alpha representing an alpha1 receptor.
    """

    def __init__(self):
        self.ki = 330
        self.offset = 5.895
        self.pretty = "α1 Receptor"
        self.gaind = -0.02
        self.biasd = 0.04
        super().__init__()


class Alpha2(Alpha):
    """
    Subclass of Alpha representing an alpha2 receptor.
    """

    def __init__(self):
        self.ki = 56
        self.offset = 1
        self.pretty = "α2 Receptor"
        self.gaind = 0.1
        self.biasd = -0.1
        super().__init__()


class Simulation():
    def __init__(self):
        self.a1 = Alpha1()
        self.a2 = Alpha2()
        self.num_spikes = np.ones(len(steps))
        self.biass = np.ones(len(steps))
        self.gains = np.ones(len(steps))
        self.out = np.ones(3)
        self.trial = 0

        self.perceived = np.ones(3)  # correctly perceived (not necessarily remembered) cues
        rng=np.random.RandomState(seed=111)
        self.cues=2 * rng.randint(2, size=3)-1  # whether the cues is on the left or right
        for n in range(len(self.perceived)):
            if rng.rand() < misperceive:
                self.perceived[n] = 0

    def plot(self):
        title = "Norepinephrine Concentration vs Spiking Rate"
        logging.info("Plotting " + title)
        plt.figure()
        plt.plot(steps, self.num_spikes)

        plt.xscale("log")

        plt.xlabel("Norepinephrine concentration (nM)")
        plt.ylabel("Spiking rate (spikes/time step)")
        plt.title(title)

        plt.draw()
        plt.savefig("./out/concentration-spiking.png")

    def cue_function(self, t):
        if t < t_cue and self.perceived[self.trial] != 0:
            return cue_scale * self.cues[self.trial]
        else:
            return 0

    def run(self):
        with nengo.Network() as net:
            # Nodes
            cue_node = nengo.Node(output=self.cue_function)
            time_node = nengo.Node(output=time_function)
            noise_wm_node = nengo.Node(output=noise_bias_function)
            noise_decision_node = nengo.Node(
                output=noise_decision_function)

            # Ensembles
            wm = nengo.Ensemble(neurons_wm, 2)
            decision = nengo.Ensemble(neurons_decide, 2)
            inputs = nengo.Ensemble(neurons_inputs, 2)
            output = nengo.Ensemble(neurons_decide, 1)

            # Connections
            nengo.Connection(cue_node, inputs[0], synapse=None)
            nengo.Connection(time_node, inputs[1], synapse=None)
            nengo.Connection(inputs, wm, synapse=tau_wm,
                             function=inputs_function)
            wm_recurrent = nengo.Connection(wm, wm, synapse=tau_wm)
            nengo.Connection(noise_wm_node, wm.neurons, synapse=tau_wm,
                             transform=np.ones((neurons_wm, 1)) * tau_wm)
            wm_to_decision = nengo.Connection(
                wm[0], decision[0], synapse=tau)
            nengo.Connection(noise_decision_node,
                             decision[1], synapse=None)
            nengo.Connection(decision, output, function=decision_function)

            # Probes
            probes_wm = nengo.Probe(wm[0], synapse=0.01)
            probe_spikes = nengo.Probe(wm.neurons)
            probe_output = nengo.Probe(output, synapse=None)

            # Run simulation
            with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
                for i, _ in tqdm(enumerate(steps), total=len(steps)):
                    wm.gain = (self.a1.gains[i] + self.a2.gains[i]) * sim.data[wm].gain
                    wm.bias = (self.a1.biass[i] + self.a2.biass[i]) * sim.data[wm].gain
                    for self.trial in range(3):
                        logging.debug(f"Simulating: trial: {self.trial}, gain: {fmt_num(self.gains[i])}, bias: {fmt_num(self.biass[i])}")
                        sim.run(t_cue + t_delay)
                        self.out[self.trial] = np.count_nonzero(sim.data[probe_spikes])
                    self.num_spikes[i] = np.average(self.out)

        with open(f"out/{datetime.now().isoformat()}-spikes.pkl", "wb") as pout:
            pickle.dump(self.num_spikes, pout)

        self.plot()

def main():
    logging.info("Initializing simulation")
    plt.style.use("ggplot")  # Nice looking and familiar style
    Simulation().run()


if __name__ == "__main__":
    try:
        mkdir("./out")
    except FileExistsError:
        pass

    logging.basicConfig(filename=f"out/{datetime.now().isoformat()}.log",
                        level=logging.DEBUG)

    main()

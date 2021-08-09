from os import mkdir
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import nengo
import numpy as np

exec(open("conf.py").read())

def fmt_num(num):
    """
    Format number to string.
    """

    return str(num)[:18].zfill(18)


def wm_recurrent_function(x):
    return x


def inputs_function(x):
    return x * tau_wm


def noise_decision_function(t):
    return np.random.normal(0.0, noise_decision)


def noise_bias_function(t):
    return np.random.normal(0.0, noise_wm)


def time_function(t):
    return time_scale if t > t_cue else 0


def decision_function(x):
    return 1.0 if x[0] + x[1] > 0.0 else -1.0


class Alpha(object):
    """
    Base class for alpha receptors. Not to be used directly.
    """

    def __init__(self):
        self.x = np.logspace(0, 3, steps)
        self.y = 1 / (1 + (999 * np.exp(-0.1233 * (self.x / self.offset))))

        self.gain = []
        self.bias = []

        for i in range(steps):
            y = self.y[i]
            self.gain.append(1 + self.gaind * y)
            self.bias.append(1 + self.biasd * y)

    def plot(self):
        try:
            mkdir("./out")
        except FileExistsError:
            pass

        out = f"./out/{self.__class__.__name__}"
        plt.figure()
        plt.plot(self.x, self.y)

        plt.xlabel("Norepinephrine concentration (nM)")
        plt.ylabel("Activity (%)")
        plt.title("Norepinepherine Concentration vs Neuron Activity in " +
                  self.pretty)

        plt.vlines(self.ki, 0, 1, linestyles="dashed")
        plt.text(1.1 * self.ki, 0.1, "Affinity")

        plt.hlines(0.5, 0, 1000, linestyles="dashed")
        plt.text(1, 0.51, "50%")

        plt.xscale("log")
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.draw()
        plt.savefig(f"{out}-norep-activity.png", dpi=1000)

        #######################################################################

        plt.figure()
        plt.plot(self.x, self.gain)

        plt.xlabel("Norepinephrine concentration (nM)")
        plt.ylabel("Gain")
        plt.title(f"Concentration vs Gain in {self.pretty}")

        plt.draw()
        plt.savefig(f"{out}-concentration-gain.png", dpi=1000)

        #######################################################################

        plt.figure()
        plt.plot(self.x, self.bias)

        plt.xlabel("Norepinephrine concentration (nM)")
        plt.ylabel("Bias")
        plt.title("Concentration vs Bias in " + self.pretty)

        plt.draw()
        plt.savefig(f"{out}-concentration-bias.png", dpi=1000)

    def simulate(self):
        for i in range(steps):
            print(f"{self.__class__.__name__}, gain: {fmt_num(self.gain[i])}, bias: {fmt_num(self.bias[i])}")
            with nengo.Network() as net:
                # Nodes
                time_node = nengo.Node(output=time_function)
                noise_wm_node = nengo.Node(output=noise_bias_function)
                noise_decision_node = nengo.Node(
                    output=noise_decision_function)

                # Ensembles
                wm = nengo.Ensemble(neurons_wm, 2)
                wm.gain = np.full(wm.n_neurons, self.gain[i])
                wm.bias = np.full(wm.n_neurons, self.bias[i])
                decision = nengo.Ensemble(neurons_decide, 2)
                inputs = nengo.Ensemble(neurons_inputs, 2)
                output = nengo.Ensemble(neurons_decide, 1)

                # Connections
                nengo.Connection(time_node, inputs[1], synapse=None)
                nengo.Connection(inputs, wm, synapse=tau_wm,
                                 function=inputs_function)
                wm_recurrent = nengo.Connection(wm, wm, synapse=tau_wm,
                                                function=wm_recurrent_function)
                nengo.Connection(noise_wm_node, wm.neurons, synapse=tau_wm,
                                 transform=np.ones((neurons_wm, 1)) * tau_wm)
                wm_to_decision = nengo.Connection(
                    wm[0], decision[0], synapse=tau)
                nengo.Connection(noise_decision_node,
                                 decision[1], synapse=None)
                nengo.Connection(decision, output, function=decision_function)

                # Probes
                # probes_wm = nengo.Probe(wm[0], synapse=0.01, sample_every=dt_sample)
                # probes_spikes = nengo.Probe(wm.neurons, 'spikes',
                #                           sample_every=dt_sample)
                # probe_output = nengo.Probe(output, synapse=None, same_every=dt_sample)

                # Run simulation
            with nengo.Simulator(net, dt=dt, progress_bar=False) as sim:
                sim.run(t_cue + t_delay)


class Alpha1(Alpha):
    """
    Subclass of Alpha representing an alpha1 receptor.
    """

    def __init__(self):
        self.ki = 330
        self.offset = 5.895
        self.pretty = "α1 Receptor"
        self.gaind = -0.04
        self.biasd = -0.02
        super().__init__()


class Alpha2(Alpha):
    """
    Subclass of Alpha representing an alpha2 receptor.
    """

    def __init__(self):
        self.ki = 56
        self.offset = 1
        self.pretty = "α2 Receptor"
        self.gaind = -0.1
        self.biasd = 0.1
        super().__init__()


def main():
    plt.style.use("ggplot")  # Nice looking and familiar style
    a1 = Alpha1()
    # a1.plot()
    a1.simulate()

    a2 = Alpha2()
    # a2.plot()
    a2.simulate()


if __name__ == "__main__":
    main()

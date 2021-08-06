#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import nengo

# Refer to parameters.txt for documentation
dt = 0.001
t_cue = 1.0
cue_scale = 1.0
perceived = 0 # ???
time_scale = 0.4
steps = 100

class Alpha(object):
    def __init__(self):
        self.x = np.logspace(0, 3, steps)
        self.y = 1 / (1 + (999 * np.exp(-0.1233 * (self.x / self.offset))))
        
        self.gain = []
        self.bias = []
        
    def calcgb(self, gaind, biasd):
        for i in range(steps):
            y = self.y[i]
            self.gain.append(1 + gaind * y)
            self.bias.append(1 + biasd * y)

    def plot(self):
        plt.plot(self.x, self.y)

        plt.xlabel("Norepinephrine concentration (nM)")
        plt.ylabel("Activity (%)")
        plt.title("Norepinepherine Concentration vs Neuron Activity in " + self.pretty)

        plt.vlines(self.ki, 0, 1, linestyles="dashed")
        plt.text(1.1 * self.ki, 0.1, "Affinity")

        plt.hlines(0.5, 0, 1000, linestyles="dashed")
        plt.text(1, 0.51, "50%")

        plt.xscale("log")
        gc = plt.gca()
        gc.set_yticklabels(['{:.0f}%'.format(x * 100) for x in gc.get_yticks()])

        plt.draw()
        plt.savefig(self.__class__.__name__ + "-norep-activity.png", dpi=1000)
        plt.show()
        
        #######################################################################
        
        plt.plot(self.x, self.gain)
        
        plt.xlabel("Norepinephrine concentration (nM)")
        plt.ylabel("Gain")
        plt.title("Concentration vs Gain in " + self.pretty)
        
        plt.draw()
        plt.savefig(self.__class__.__name__ + "-concentration-gain.png", dpi=1000)
        plt.show()
        
        #######################################################################
        
        plt.plot(self.x, self.bias)
        
        plt.xlabel("Norepinephrine concentration (nM)")
        plt.ylabel("Bias")
        plt.title("Concentration vs Bias in " + self.pretty)
        
        plt.draw()
        plt.savefig(self.__class__.__name__ + "-concentration-bias.png", dpi=1000)
        plt.show()

class Alpha1(Alpha):
    def __init__(self):
        self.ki = 330
        self.offset = 5.895
        self.pretty = u"α1 Receptor"
        self.gaind = 0.1
        self.biasd = 0.1
        super().__init__()
        
    def calcgb(self):
        super().calcgb(self.gaind, self.biasd)

class Alpha2(Alpha):
    def __init__(self):
        self.ki = 56
        self.offset = 1
        self.pretty = u"α2 Receptor"
        self.gaind = -0.04
        self.biasd = -0.02
        super().__init__()
        
    def calcgb(self):
        super().calcgb(self.gaind, self.biasd)

def main():
    plt.style.use("ggplot")
    
    a1 = Alpha1()
    a1.calcgb()
    a1.plot()

    a2 = Alpha2()
    a2.calcgb()
    a2.plot()

if __name__ == "__main__":
    main()

#!/usr/bin/env python

from matplotlib import pyplot as plt
import network, agent
import numpy as np

parameters = {"innoise" : 2, # Stddev on incomming messages
              "outnoise" : 2, # Stddev on outgoing messages
              "num_environment" : 5, # Num univariate environment nodes
              "num_agents" : 10, # Number of Agents
              "fanout" : 1, # Distinct messages an agent can say
              "statedim" : 1, # Dimension of Agent State
              "envnoise": 25, # Stddev of environment state
              "envobsnoise" : 2, # Stddev on observing environment
              "batchsize" : 1000} # Training Batch Size

if __name__ == "__main__":
    plt.ion()
    org = network.Organization(**parameters)
    res = org.train(3000, 100, iplot=False)
    fix,ax = plt.subplots()
    ax.plot(np.log(res.training_res), label="Experiment 1")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Log(Welfare")
    ax.legend()
    G = res.graph_org()

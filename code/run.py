#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import network, agent
import numpy as np

parameters = []

parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 1000, # Training Batch Size
    "description" : "Baseline"}
)

parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 5, # Stddev on observing environment
    "batchsize" : 1000, # Training Batch Size
    "description" : "Environment Expensive"}
)

parameters.append(
    {"innoise" : 10, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 1000, # Training Batch Size
    "desciption" : "Messages Expensive"}
)

parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 10, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 1000, # Training Batch Size
    "description" : "Double Environment"}
)

parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 20, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 1000, # Training Batch Size
    "description" : "Double Agents"}
)

parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 10000, # Training Batch Size
    "description" : "10x Batch Size"}
)

# 7: Baseline, but with 1/10th batch size
parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 100, # Training Batch Size
    "description" : "1/10th Batch Size"}
)

if __name__ == "__main__":
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    res = None
    iterations = 3000
    for i in range(len(parameters)):
        p = parameters[i]
        print "Running trial %d (%s)" % (i+1, p["description"])
        print " * Initializing network 1"
        orgA = network.Organization(optimizer="adadelta", **p)
        print " * Training network 1"
        resA = orgA.train(iterations, iplot=False, verbose=False)
        print " * Initializing network 2"
        orgB = network.Organization(optimizer="rmsprop", **p)
        print " * Training network 2"
        resB = orgB.train(iterations, iplot=False, verbose=False)
        print " * Saving better network"
        if( resA.welfare > resB.welfare ):
            res = resA
        else:
            res = resB
        ax.plot(np.log(res.training_res), label=p["description"])
        res.graph_cytoscape("trial" + str(i+1) + ".graphml")
    ax.set_title("Trials")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Log(Welfare)")
    ax.legend()
    fig.savefig("trials.png")

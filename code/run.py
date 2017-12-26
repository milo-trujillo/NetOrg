#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import network, agent
import numpy as np
import pickle
import multiprocessing
import copy

'''
    This file defines what simulations will run, and for how long.
    It can be run as a standalone script ('./run.py') or used as the job
    target on SFI's cluster system.

    Below are a number of predefined simulation parameters. We usually
    either run through all predefined simulations to test the network,
    or in the case of parameter sweeps, start with the baseline simulation
    and scale individual parameters as needed.
'''

parameters = []

# Trivial network: 1 agent, no managers, 5 env nodes
parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 1, # Number of Agents
    "num_managers" : 0, # Number of Agents that do not contribute
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state (NO LONGER USED)
    "envobsnoise" : 1, # Stddev on observing environment
    "batchsize" : 1000, # Training Batch Size
    "layers"      : 1, # Number of layers per agent
    "description" : "Baseline"}
)

# A more standard network: 3 agents, 2 of whom aremanagers, 5 env
parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 3, # Number of Agents
    "num_managers" : 2, # Number of Agents that do not contribute
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 0.0000001, # Stddev on observing environment
    "batchsize" : 1000, # Training Batch Size
    "layers"      : 1, # Number of layers per agent
    "description" : "Three Agents"}
)

# Bigger network: 10 agents (9 managers), 10 env
parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 10, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "num_managers" : 9, # Number of Agents that do not contribute
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 100, # Training Batch Size
    "layers"      : 3, # Number of layers per agent
    "description" : "Double Environment"}
)

# More agents, small network (tests how we handle redundancy)
# 20 agents, 5 env
parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 20, # Number of Agents
    "num_managers" : 19, # Number of Agents that do not contribute
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 100, # Training Batch Size
    "layers"      : 3, # Number of layers per agent
    "description" : "Double Agents"}
)

# This runs a single simulation for specified number of iterations
# It tests with two optimizers (adadelta and rmsprop) and saves the better
# result, in case one optimizer gets stuck in a local minima
def runSim(parameters, iteration, iterations):
    print "Running trial %d (%s)" % (iteration+1, parameters["description"])
    print " * Initializing network 1"
    orgA = network.Organization(optimizer="adadelta", **parameters)
    print " * Training network 1"
    resA = orgA.train(iterations, iplot=False, verbose=False)
    print " * Initializing network 2"
    orgB = network.Organization(optimizer="rmsprop", **parameters)
    print " * Training network 2"
    resB = orgB.train(iterations, iplot=False, verbose=False)
    if( resA.welfare < resB.welfare ):
        res = resA
    else:
        res = resB
    print " * Saving better network (Welfare %f)" % res.welfare
    return res

# This runs a simulation several times, with restarts in case
# both optimizers get stuck in a local minima
# Saves best result to disk as pickled data
def runIterations(parameters, restarts, numIterations, filename):
    res = None
    for restart in range(restarts):
        result = runSim(parameters, restart, numIterations)
        if( res == None or result.welfare < res.welfare ):
            res = result
    pickle.dump(res, open(filename + "_res.pickle", "wb"))

# This code block defines which simulations we will run, for how long,
# and what results to graph
if __name__ == "__main__":
    plt.ion()
    welfarefig = plt.figure()
    welfareax = welfarefig.add_subplot(1,1,1)
    res = None
    iterations = 3000

    xs = []
    costYs = []
    diffYs = []
    wellYs = []

    # To run all simulations use
    #for i in range(len(parameters)):
    #   p = copy.deepcopy(parameters[i])

    # For a parameter sweep, do something like
    #for env in range(160):
        #p = copy.deepcopy(parameters[1])
        #p["num_environment"] = 5 + env

    # Currently we run a single small test to verify code functionality
    for env in range(1):
        p = copy.deepcopy(parameters[1])
        p["num_environment"] = 5 + env
        p["description"] = str(p["num_environment"])
        filename = "trial_%s" % (p["description"])

        # NOTE: We run all simulations on background processes.
        # This is because Tensorflow does not release its memory after we
        # finish running a network, so if we run many simulations in one process
        # we'll swell to using 200GB of memory. This way memory is forcibly
        # freed by process termination after each simulation.
        proc = multiprocessing.Process(target=runIterations, args=(p, 1, iterations, filename,))
        proc.start()
        proc.join()

        # Disable the 'proc' lines and uncommend the following to run everything
        # in the main process for easier debugging
        #runIterations(p, 3, iterations, filename)

        # Read results, add them to the graph, and export them in a format
        # Cytoscape and Gephi can render as pretty network diagrams
        res = pickle.load(open(filename + "_res.pickle", "rb"))
        filename = "trial_%s_welfare_%f" % (p["description"], res.welfare)
        res.graph_cytoscape(filename + ".gml")
        res.graph_collapsed_cytoscape(filename + "_collapsed.gml")
        welfareax.plot(res.training_res, label=p["description"])
    welfareax.set_title("Trials")
    welfareax.set_xlabel("Training Epoch")
    welfareax.set_ylabel("Welfare")
    welfareax.legend()
    welfarefig.savefig("trials.png")

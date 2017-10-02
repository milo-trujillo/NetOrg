#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import network, agent
import numpy as np
import pickle
import multiprocessing
import copy

parameters = []

parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 6, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "num_managers" : 9, # Number of Agents that do not contribute
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 1000, # Training Batch Size
    "layers"      : 1, # Number of layers per agent
    "description" : "Baseline"}
)

parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "num_managers" : 9, # Number of Agents that do not contribute
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 5, # Stddev on observing environment
    "batchsize" : 100, # Training Batch Size
    "layers"      : 3, # Number of layers per agent
    "description" : "Environment Expensive"}
)

parameters.append(
    {"innoise" : 10, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "num_managers" : 9, # Number of Agents that do not contribute
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 100, # Training Batch Size
    "layers"      : 3, # Number of layers per agent
    "description" : "Messages Expensive"}
)

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

def runSim(parameters, iteration, iterations):
    print "Running trial %d (%s)" % (iteration+1, parameters["description"])
    print " * Initializing network 1"
    orgA = network.Organization(optimizer="adadelta", **parameters)
    print " * Training network 1"
    resA = orgA.train(iterations, iplot=False, verbose=True)
    print " * Initializing network 2"
    orgB = network.Organization(optimizer="rmsprop", **parameters)
    print " * Training network 2"
    resB = orgB.train(iterations, iplot=False, verbose=True)
    if( resA.welfare < resB.welfare ):
        res = resA
    else:
        res = resB
    print " * Saving better network (Welfare %f)" % res.welfare
    return res

def runIterations(parameters, restarts, numIterations, filename):
    res = None
    for restart in range(restarts):
        result = runSim(parameters, restart, numIterations)
        if( res == None or result.welfare < res.welfare ):
            res = result
    pickle.dump(res, open(filename + "_res.pickle", "wb"))

if __name__ == "__main__":
    plt.ion()
    welfarefig = plt.figure()
    welfareax = welfarefig.add_subplot(1,1,1)
    resultsfig = plt.figure()
    resultsax = resultsfig.add_subplot(1,1,1)
    res = None
    iterations = 3000

    xs = []
    costYs = []
    diffYs = []
    wellYs = []

    #for i in range(len(parameters)):
    for i in range(40):
        p = copy.deepcopy(parameters[0])
        p["num_agents"] += i
        filename = "trial%d" % (i+1)
        #proc = multiprocessing.Process(target=runIterations, args=(p, 3, iterations, filename,))
        #proc.start()
        #proc.join()
        runIterations(p, 3, iterations, filename)
        res = pickle.load(open(filename + "_res.pickle", "rb"))
        filename = "trial%d_welfare_%f" % (i+1, res.welfare)
        res.graph_cytoscape(filename + ".gml")
        res.graph_collapsed_cytoscape(filename + "_collapsed.gml")
        welfareax.plot(np.log(res.training_res), label=p["description"])
        xs += [p["num_agents"]]
        costYs += [res.welfareCost]
        diffYs += [res.welfareDifference]
    resultsax.plot(xs, costYs, label="Communication Cost")
    resultsax.plot(xs, diffYs, label="Difference from Optimum")
    welfareax.set_title("Trials")
    welfareax.set_xlabel("Training Epoch")
    welfareax.set_ylabel("Log(Welfare)")
    welfareax.legend()
    welfarefig.savefig("trials.png")
    resultsax.set_title("Trials")
    resultsax.set_xlabel("Agents")
    resultsax.set_ylabel("Welfare")
    resultsax.legend()
    resultsfig.savefig("trial_results.png")

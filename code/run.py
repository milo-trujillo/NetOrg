#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import network, agent
import numpy as np
import pickle
import copy
import multiprocessing

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
    "batchsize" : 100, # Training Batch Size
    "layers"      : 3, # Number of layers per agent
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
    "batchsize" : 100, # Training Batch Size
    "layers"      : 3, # Number of layers per agent
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
    "batchsize" : 100, # Training Batch Size
    "layers"      : 3, # Number of layers per agent
    "description" : "Messages Expensive"}
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
    "batchsize" : 100, # Training Batch Size
    "layers"      : 3, # Number of layers per agent
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
    "batchsize" : 100, # Training Batch Size
    "layers"      : 3, # Number of layers per agent
    "description" : "Double Agents"}
)

def runSim(parameters, iteration):
    print "Running trial %d (%s)" % (iteration+1, parameters["description"])
    print " * Initializing network 1"
    orgA = network.Organization(optimizer="adadelta", randomSeed=True, **p)
    print " * Training network 1"
    resA = orgA.train(iterations, iplot=False, verbose=False)
    print " * Initializing network 2"
    orgB = network.Organization(optimizer="rmsprop", randomSeed=True, **p)
    print " * Training network 2"
    resB = orgB.train(iterations, iplot=False, verbose=False)
    if( resA.welfare < resB.welfare ):
        res = resA
    else:
        res = resB
    print " * Saving better network (Welfare %f)" % res.welfare
    return res

# Tensorflow is really bad about freeing memory, so we get around
# the problem by running batches of tensorflow simulations in subprocesses.
# Killing the subprocesses forcibly frees the memory, and keeps us from
# using 200 GB on longer runs
def runIterations(parameters, numIterations, filename):
    res = None
    for restart in range(10):
        result = runSim(parameters, restart)
        if( res == None or result.welfare < res.welfare ):
            res = result
    pickle.dump(res, open(filename + "_res.pickle", "wb"))

if __name__ == "__main__":
    plt.ion()
    welfarefig = plt.figure()
    welfareax = welfarefig.add_subplot(1,1,1)
    listencostfig = plt.figure()
    listencostax = listencostfig.add_subplot(1,1,1)
    degreefig = plt.figure()
    degreeax = degreefig.add_subplot(1,1,1)
    iterations = 1000
    for i in range(10):
        p = copy.deepcopy(parameters[0])
        p["innoise"] += i
        filename = "trial%d" % (i+1)
        proc = multiprocessing.Process(target=runIterations, args=(p, 10, filename,))
        proc.start()
        proc.join()
        res = pickle.load(open(filename + "_res.pickle", "rb"))
	    res.graph_cytoscape(filename + "_welfare_" + str(res.welfare) + ".gml")
	    res.graph_collapsed_cytoscape(filename + "_welfare_" + str(res.welfare) + "_collapsed.gml")
        welfareax.plot(np.log(res.training_res), label=p["description"])
        listencostax.plot([p["innoise"]], [res.global_reaching_centrality()])
        degreeax.plot([p["innoise"]], [res.get_degree_distribution()])
    welfareax.set_title("Trials")
    welfareax.set_xlabel("Training Epoch")
    welfareax.set_ylabel("Log(Welfare)")
    welfareax.legend()
    welfarefig.savefig("trials.png")
    listencostax.set_title("Centrality Parameter Sweep")
    listencostax.set_xlabel("Listen Cost")
    listencostax.set_ylabel("Global Reaching Centrality")
    listencostax.legend()
    listencostfig.savefig("centrality_sweep.png")
    degreeax.set_title("Centrality Parameter Sweep")
    degreeax.set_xlabel("Listen Cost")
    degreeax.set_ylabel("Degree Std-Deviation (per agent)")
    degreeax.legend()
    degreefig.savefig("degree_sweep.png")

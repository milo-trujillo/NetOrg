#!/usr/bin/env python

from matplotlib import pyplot as plt
import network, agent
import numpy as np

parameters = []

# 1: Baseline Justin used
parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 1000} # Training Batch Size
)

# 2: Listening to environment is extra expensive
parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 5, # Stddev on observing environment
    "batchsize" : 1000} # Training Batch Size
)

# 3: Listening to messages is extra expensive
parameters.append(
    {"innoise" : 10, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 1000} # Training Batch Size
)

# 4: There are twice as many environment nodes to listen to
parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 10, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 1000} # Training Batch Size
)

# 5: There are twice as many agent nodes as normal
parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 20, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 1000} # Training Batch Size
)

# 6: Baseline, but with 10x batch size
parameters.append(
    {"innoise" : 2, # Stddev on incomming messages
    "outnoise" : 2, # Stddev on outgoing messages
    "num_environment" : 5, # Num univariate environment nodes
    "num_agents" : 10, # Number of Agents
    "fanout" : 1, # Distinct messages an agent can say
    "statedim" : 1, # Dimension of Agent State
    "envnoise": 25, # Stddev of environment state
    "envobsnoise" : 2, # Stddev on observing environment
    "batchsize" : 10000} # Training Batch Size
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
    "batchsize" : 100} # Training Batch Size
)

if __name__ == "__main__":
    plt.ion()
    orgs = []
    results = []
    fix,ax = plt.subplots()
    #for i in range(len(parameters)):
    for i in ([0]):
        print "Running trial %d" % (i+1)
        p = parameters[i]
        org = network.Organization(optimizer="adadelta", **p)
        orgs.append(org)
        res = org.train(3000, 100, iplot=False, verbose=True)
        results.append(res)
    	ax.plot(np.log(res.training_res), label="Experiment "+str(i+1))
    	res.graph_cytoscape("trial" + str(i+1) + ".graphml")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Log(Welfare)")
    ax.legend()
    #plt.show(block=True)
    plt.savefig("trial.png")
    #G = res.graph_org()

#!/usr/bin/env python

from matplotlib import pyplot as plt
import network, agent
import numpy as np

# Baseline Justin used
parameters1 = {"innoise" : 2, # Stddev on incomming messages
              "outnoise" : 2, # Stddev on outgoing messages
              "num_environment" : 5, # Num univariate environment nodes
              "num_agents" : 10, # Number of Agents
              "fanout" : 1, # Distinct messages an agent can say
              "statedim" : 1, # Dimension of Agent State
              "envnoise": 25, # Stddev of environment state
              "envobsnoise" : 2, # Stddev on observing environment
              "batchsize" : 1000} # Training Batch Size

# Listening to environment is extra expensive
parameters2 = {"innoise" : 2, # Stddev on incomming messages
              "outnoise" : 2, # Stddev on outgoing messages
              "num_environment" : 5, # Num univariate environment nodes
              "num_agents" : 10, # Number of Agents
              "fanout" : 1, # Distinct messages an agent can say
              "statedim" : 1, # Dimension of Agent State
              "envnoise": 25, # Stddev of environment state
              "envobsnoise" : 5, # Stddev on observing environment
              "batchsize" : 1000} # Training Batch Size

# Listening to messages is extra expensive
parameters3 = {"innoise" : 10, # Stddev on incomming messages
              "outnoise" : 2, # Stddev on outgoing messages
              "num_environment" : 5, # Num univariate environment nodes
              "num_agents" : 10, # Number of Agents
              "fanout" : 1, # Distinct messages an agent can say
              "statedim" : 1, # Dimension of Agent State
              "envnoise": 25, # Stddev of environment state
              "envobsnoise" : 2, # Stddev on observing environment
              "batchsize" : 1000} # Training Batch Size

# There are twice as many environment nodes to listen to
parameters4 = {"innoise" : 2, # Stddev on incomming messages
              "outnoise" : 2, # Stddev on outgoing messages
              "num_environment" : 10, # Num univariate environment nodes
              "num_agents" : 10, # Number of Agents
              "fanout" : 1, # Distinct messages an agent can say
              "statedim" : 1, # Dimension of Agent State
              "envnoise": 25, # Stddev of environment state
              "envobsnoise" : 2, # Stddev on observing environment
              "batchsize" : 1000} # Training Batch Size

# There are twice as many agent nodes as normal
parameters5 = {"innoise" : 2, # Stddev on incomming messages
              "outnoise" : 2, # Stddev on outgoing messages
              "num_environment" : 5, # Num univariate environment nodes
              "num_agents" : 20, # Number of Agents
              "fanout" : 1, # Distinct messages an agent can say
              "statedim" : 1, # Dimension of Agent State
              "envnoise": 25, # Stddev of environment state
              "envobsnoise" : 2, # Stddev on observing environment
              "batchsize" : 1000} # Training Batch Size

parameters = [parameters1, parameters2, parameters3, parameters4, parameters5]

if __name__ == "__main__":
    plt.ion()
    orgs = []
    results = []
    fix,ax = plt.subplots()
    #for i in ([0]):
    for i in range(len(parameters)):
        p = parameters[i]
        org = network.Organization(**p)
        orgs.append(org)
        res = org.train(3000, 100, iplot=False)
        results.append(res)
    	ax.plot(np.log(res.training_res), label="Experiment "+str(i+1))
    	res.graph_cytoscape("trial" + str(i+1) + ".graphml")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Log(Welfare)")
    ax.legend()
    plt.show(block=True)
    #G = res.graph_org()

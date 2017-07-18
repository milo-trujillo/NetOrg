#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from agent import Agent
import copy
import matplotlib as mpl
mpl.use('Agg') # Don't require a GUI to generate graphs
from matplotlib import pyplot as plt
import time
import os
import networkx as nx
plt.ion()

class Organization(object):
    def __init__(self, num_environment, num_agents, innoise,
                     outnoise, fanout, statedim, envnoise, envobsnoise,
                     batchsize, optimizer, layers, randomSeed=False, tensorboard=None, **kwargs):
        tf.reset_default_graph() # Clear all existing tensorflow memory
        if( randomSeed == False ):
            tf.set_random_seed(634)
        self.sess = tf.Session()
        self.num_environment = num_environment
        self.num_agents = num_agents
        self.batchsize = batchsize
        self.envobsnoise = envobsnoise
        self.layers = layers
        self.agents = []
        for i in range(num_agents * layers):
            self.agents.append(Agent(innoise, outnoise, i, fanout, statedim, batchsize, num_agents, num_environment))
        self.environment = tf.random_normal([self.batchsize, num_environment],
                                            mean=0, stddev = envnoise, dtype=tf.float64)
        self.build_org()
        self.objective  =  self.ruggedLoss()
        self.learning_rate = tf.placeholder(tf.float64)

        # Justin used the "AdadeltaOptimizer"
        optimizers = {
            "momentum":         tf.train.MomentumOptimizer(self.learning_rate, momentum=0.5).minimize(self.objective),
            "adadelta":         tf.train.AdadeltaOptimizer(self.learning_rate, rho=.9).minimize(self.objective),
            "adam":             tf.train.AdamOptimizer(self.learning_rate).minimize(self.objective),
            "rmsprop":          tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.objective),
            "gradient-descent": tf.train.AdagradOptimizer(self.learning_rate).minimize(self.objective)
            }

        learning_rates = {
            "momentum":         1e-6,
            "adadelta":         15,
            "adam":             1e-2,
            "rmsprop":          1e-2,
            "gradient-descent": 1e-1
        }

        decays = {
            "momentum":         None,
            "adadelta":         0.001,
            "adam":             0.001,
            "rmsprop":          0.001,
            "gradient-descent": None
        }

        self.optimize = optimizers[optimizer]
        self.start_learning_rate = learning_rates[optimizer]
        self.decay = decays[optimizer]

        if( tensorboard == None ):
            self.writer = None
        else:
            self.writer = tf.summary.FileWriter(tensorboard, self.sess.graph)
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def build_org(self):
        self.build_agent_params()
        self.build_wave()

    def build_agent_params(self):
        indim = self.num_environment
        created = []
        for ix, a in enumerate(self.agents):
            #print "Agent %d gets indim=%d" % (ix, indim)
            created.append(a)
            if( ix < self.num_agents ):
                a.create_in_vec(indim)
                a.create_state_matrix(indim)
                a.create_out_matrix(indim)
            else:
                a.set_predecessor(created.pop(0))
                a.create_in_vec(self.num_agents)
                a.create_state_matrix(self.num_agents + a.predecessor.indim)
                a.create_out_matrix(self.num_agents + a.predecessor.indim)
            indim += a.fanout

    def build_wave(self):
        """
        This loops through agents building the tensorflow objects
        that determine the agents states and outputs that are then
        recursively use to build all other agent's states and outputs
        """
        self.states = []
        self.outputs = [[] for x in range(0, len(self.agents)+1)]
        for a in self.agents:
            envnoise = tf.random_normal([self.batchsize, self.num_environment], stddev=self.envobsnoise, dtype=tf.float64)

            noisyin = []
            state = []
            a.state = []

            for i in range(0, self.num_agents + 1):
                inenv = self.environment
                incomm = None #?

                # First wave
                if( a.predecessor == None ):
                    for inmsgs in self.outputs[i]:
                        if incomm is None:
                            incomm = inmsgs # Stays None if inmsgs blank, otherwise becomes inmsgs
                        else:
                            incomm =  tf.concat([incomm, inmsgs], 1) # If already a message, then concat
                    commnoise = tf.random_normal([self.batchsize, a.indim - self.num_environment], stddev=a.noiseinstd, dtype=tf.float64)
                    # Noise on inputs
                    if incomm is not None:
                        indata = tf.concat([inenv, incomm], 1) # batchsize x 
                    else:
                        indata = inenv
                    innoise = tf.concat([envnoise, commnoise], 1)
                # Second wave and up
                else:
                    '''
                    We're only listening to the first wave nodes (no environment),
                    so can skip all the inenv and envnoise steps
                    '''
                    loadLayerStart = ((a.num / self.num_agents) - 1) * self.num_agents
                    loadLayerEnd = (a.num / self.num_agents) * self.num_agents
                    indata = tf.concat(self.outputs[i][loadLayerStart:loadLayerEnd], 1)
                    commnoise = tf.random_normal([self.batchsize, self.num_agents], stddev=a.noiseinstd, dtype=tf.float64)
                    innoise = commnoise

                # Add noise inversely-proportional to listening strength
                noisyin.append(indata + innoise/a.listen_weights[i])
                if( a.predecessor != None ):
                    noisyin[i] = tf.concat([a.predecessor.received_messages[i], noisyin[i]], 1)
                a.set_received_messages(i, noisyin[i])

                # Since listen weights is 1xin we get row wise division.
                state.append(tf.matmul(noisyin[i], a.state_weights[i]))
                a.state.append(state[i])
                self.states.append(state[i])

                outnoise = tf.random_normal([self.batchsize, a.fanout], stddev=a.noiseoutstd, dtype=tf.float64)
                #outnoise = tf.random_uniform([self.batchsize, a.fanout], minval=a.noiseoutstd,maxval=a.noiseoutstd, dtype=tf.float64)
                prenoise = tf.matmul(noisyin[i], a.out_weights[i])
                output = prenoise + outnoise
                self.outputs[i].append(output)

    def listening_cost(self, exponent=2, iteration=0):
        summed = [tf.reduce_sum(tf.abs(x.listen_weights[iteration]))**exponent for x in self.agents]
        totalc = tf.add_n(summed)
        return totalc

    def speaking_cost(self, exponent=2, iteration=0):
        summed = [tf.reduce_sum(tf.abs(x.out_weights[iteration]))**exponent for x in self.agents]
        totalc = tf.add_n(summed)
        return totalc

    # Gets avg loss, if we turn off one node at a time (should emphasize redundancy)
    # This is tricky, since we have to describe it functionally, not iteratively,
    # since tensorflow variables are evaluated at a later point
    def ruggedLoss(self, exponent=2):
        differences = []
        costs = []
        realValue = tf.reduce_mean(self.environment, 1, keep_dims=True)
        lastLayer = self.num_agents * (self.layers - 1)
        for i in range(0, self.num_agents + 1): # For each layer
            diff = [tf.reduce_mean((realValue - a.state[i]) ** exponent) for a in self.agents[lastLayer:]]
            diffSum = tf.add_n(diff)
            differences.append(diffSum)
            c = self.listening_cost(iteration=i) + self.speaking_cost(iteration=i)
            costs.append(c)
        differenceSum = tf.add_n(differences)
        cost = tf.add_n(costs)
        loss = tf.add(differenceSum, cost)
        return loss

    def train(self, niters, lrinit=None, iplot=False, verbose=False):
        if( lrinit == None ):
            lrinit = self.start_learning_rate
        if iplot:
            fig, ax = plt.subplots()
            ax.plot([1],[1])
            ax.set_xlim(0,niters)
            ax.set_ylim(0,10)
            ax.set_ylabel("Welfare (Log)")
            ax.set_xlabel("Training Epoch")
            line = ax.lines[0]
        training_res = []

        # For each iteration
        for i  in range(niters):

            # Run training, and adjust learning rate if it's an Optimizer that
            # works with decaying learning rates (some don't)
            lr = float(lrinit)
            if( self.decay != None ):
                lr = float(lrinit) / (1 + i*self.decay) # Learn less over time
            self.sess.run(self.optimize, feed_dict={self.learning_rate:lr})


            #for a in self.agents:
                #a.normalize()
            listen_params = self.sess.run([a.listen_weights[0] for a in self.agents])
            output_params = self.sess.run([a.out_weights[0] for a in self.agents])
            if verbose:
                print "Listen_params now set to: " + str(listen_params[0])
                print "Output_params now set to: " + str(output_params[0])

            # Prints the agent's current strategy at each step so we can see how well it's doing
            #strat = self.sess.run(self.agents[0].listen_weights)
            #print(strat)

            # Evaluates our current progress towards objective
            u = self.sess.run(self.objective)
            if verbose:
                print  "Loss function=" + str(u)
            training_res.append(u)

            if (i%50==0) and iplot:
                line.set_data(np.arange(len(training_res)), np.log(training_res))
                fig.canvas.draw()

        # Get the strategy from all agents, which is the "network configuration" at the end
        listen_params = self.sess.run([a.listen_weights[0] for a in self.agents])
        welfare = self.sess.run(self.objective)
        if( verbose ):
            print "Listen_params now set to: " + str(listen_params)
        if( self.writer != None ):
            self.writer.close()
        return Results(training_res, listen_params, self.num_agents, self.num_environment, welfare)
    
class Results(object):
    def __init__(self, training_res, listen_params, num_agents, num_env, welfare):
        self.training_res = training_res
        self.listen_params = listen_params
        self.get_trimmed_listen_params()
        self.welfare = welfare
        self.num_agents = num_agents
        self.num_env = num_env

    def get_trimmed_listen_params(self, cutoff=.1):
        self.trimmed = []
        for lparams in self.listen_params:
            maxp = np.max(lparams)
            lparams = lparams * np.int_(lparams * lparams>cutoff*maxp)
            self.trimmed.append(lparams)

    def graph_cytoscape(self, filename, vspace=1, hspace=2):
        numenv = len(self.trimmed[0].flatten())
        G = nx.DiGraph()
        for i in range(numenv):
            G.add_node(i, color="b", name="E" + str(i), category="environment")
        for aix, agent in enumerate(self.trimmed):
            nodenum = int(numenv + aix)
            prefix = aix % self.num_agents
            layer = aix / self.num_agents
            n = "A%d_%d" % (prefix, layer)
            G.add_node(nodenum, color='r', name=n, category="agent")
            # For each node, weights will be zero if the edge should be ignored
            # and otherwise represent the cost of the edge
            for dest, weight in enumerate(agent.flatten()):
                # We need to offset the other end of the arrow, because agents
                # after layer0 don't listen to env, and only listen to the layer
                # immediately below them
                if( layer > 0 ):
                    dest += self.num_env
                    dest += (self.num_agents * (layer-1))
                if( abs(weight) > 0 ):
                    G.add_edge(int(dest), nodenum, width=float(weight), weight=float(abs(weight)))
        nx.write_graphml(G, filename)
        #nx.write_gml(G, filename)
            
    def _get_pos(self, G):
        numenv = len(self.trimmed[0].flatten())
        numnodes = numenv + len(self.trimmed)

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
        if( randomSeed == False ):
            tf.set_random_seed(634)
        self.num_environment = num_environment
        self.num_agents = num_agents
        self.batchsize = batchsize
        self.envobsnoise = envobsnoise
        self.layers = layers
        self.agents = []
        for i in range(num_agents * self.layers):
            self.agents.append(Agent(innoise, outnoise, i, fanout, statedim, batchsize, num_agents, num_environment))
        self.environment = tf.random_normal([self.batchsize, num_environment],
                                            mean=0, stddev = envnoise, dtype=tf.float64)
        self.build_org()
        self.objective  =  self.loss()
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

        self.sess = tf.Session()
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
        created = []
        for i, a in enumerate(self.agents):
            created.append(a)
            # First wave
            if( i < self.num_agents ):
                a.create_in_vec(self.num_environment)
                a.create_state_matrix(self.num_environment)
                a.create_out_matrix(self.num_environment)
            # Second wave and up
            else:
                old_version = created.pop(0)
                a.set_predecessor(old_version)
                a.create_in_vec(self.num_agents)
                a.create_state_matrix(self.num_agents + old_version.indim)
                a.create_out_matrix(self.num_agents + old_version.indim)

    def build_wave(self):
        """
        This loops through agents building the tensorflow objects
        that determine the agents states and outputs that are then
        recursively use to build all other agent's states and outputs
        """
        self.states = []
        self.outputs = []
        for i, a in enumerate(self.agents):
            envnoise = tf.random_normal([self.batchsize, self.num_environment], stddev=self.envobsnoise, dtype=tf.float64)
            inenv = self.environment

            incomm = None

            # First wave
            if( a.predecessor == None ):
                indata = inenv
                innoise = envnoise
            # Second wave+
            else:
                '''
                We're only listening to the first wave nodes (no environment),
                so can skip all the inenv and envnoise steps
                '''
                loadLayerStart = ((a.num / self.num_agents) - 1) * self.num_agents
                loadLayerEnd = (a.num / self.num_agents) * self.num_agents
                indata = tf.concat(self.outputs[loadLayerStart:loadLayerEnd], 1)
                commnoise = tf.random_normal([self.batchsize, self.num_agents], stddev=a.noiseinstd, dtype=tf.float64)
                innoise = commnoise

            # Add noise inversely-proportional to listening strength
            noisyin = indata + innoise/a.listen_weights

            # Since listen weights is 1xin we get row wise division.
            if( a.predecessor != None ):
                noisyin = tf.concat([a.predecessor.received_messages, noisyin], 1)
            a.set_received_messages(noisyin)

            state = tf.matmul(noisyin, a.state_weights)
            a.state = state
            self.states.append(state)

            outnoise = tf.random_normal([self.batchsize, a.fanout], stddev=a.noiseoutstd, dtype=tf.float64)
            prenoise = tf.matmul(noisyin, a.out_weights)
            output = prenoise + outnoise
            self.outputs.append(output)
            # output is a vector with dimensions [1, batchsize]
            #with tf.Session() as sess:
                #init = tf.global_variables_initializer()
                #sess.run(init)
                #res = sess.run(output)
                #print "Appending output for agent " + str(i) + ": " + str(res)

    def listening_cost(self, exponent=2, env_exponent=3):
        lastLayer = self.num_agents * (self.layers - 1)
        summed = [x.listen_cost(exponent, env_exponent) for x in self.agents[lastLayer:]]
        totalc = tf.add_n(summed)
        return totalc

    def speaking_cost(self, exponent=2):
        lastLayer = self.num_agents * (self.layers - 1)
        summed = [x.speaking_cost(exponent) for x in self.agents[lastLayer:]]
        totalc = tf.add_n(summed)
        return totalc

    '''
    In fractured_goals, each node is only responsible for knowing about a subset of the environment
    Specifically, each node is responsible for n%env, n+1%env, and n+2%env
    '''
    def loss(self, exponent=2):
        lastLayer = self.num_agents * (self.layers - 1)
        env = self.num_environment
        differences = []
        for a in self.agents[lastLayer:]:
            goals = [a.num % env, (a.num + 1) % env, (a.num + 2) % env]
            realValue = tf.reduce_mean(tf.gather(self.environment, goals))
            differences.append(tf.reduce_mean(realValue - a.state)**exponent)
        differenceSum = tf.add_n(differences)
        cost = self.listening_cost() + self.speaking_cost()
        loss = differenceSum + cost
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

            if verbose:
                listen_params = self.sess.run([a.listen_weights for a in self.agents])
                output_params = self.sess.run([a.out_weights for a in self.agents])
                print "Listen_params now set to: " + str(listen_params)
                #print "Output_params now set to: " + str(output_params)

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
        listen_params = self.sess.run([a.listen_weights for a in self.agents])
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

    def generate_graph(self, vspace=1, hspace=2):
        numenv = len(self.trimmed[0].flatten())
        numnodes = numenv + len(self.trimmed)
        G = nx.DiGraph()
        vpos = 0
        hpos = 0
        for i in range(numenv):
            G.add_node(i, color="b", name="E" + str(i), pos=(hpos, vpos))
            hpos += hspace
        vpos = vspace
        hpos = 0
        hspace = hspace*numenv/float(len(self.trimmed))
        highest_listened = numenv - 1
        for aix, agent in enumerate(self.trimmed):
            hpos += hspace
            nextlevel = False
            nodenum = numenv +aix
            G.add_node(nodenum, color='r', name = "A" + str(aix))
            for eix, val in enumerate(agent.flatten()):
                if abs(val) > 0:
                    G.add_edge(eix, nodenum, width=val)
                    if eix > highest_listened:
                        highest_listened =eix
                        nextlevel=True
            if nextlevel:
                vpos += vspace
            G.node[nodenum]["pos"] = (hpos, vpos)
        return G

    def graph_org(self, vspace=1, hspace=2):
        G = self.generate_graph(vspace, hspace)
        colors = nx.get_node_attributes(G, "color").values()
        pos= nx.get_node_attributes(G, "pos")
        nx.draw(G, pos, node_color = colors, with_labels=True,
                    labels=nx.get_node_attributes(G, "name"), alpha=.5, node_size=600 )
        return G

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
            if( layer > 0 ):
                predecessor = int(numenv + aix - self.num_agents)
                G.add_edge(predecessor, nodenum, width=0, weight=0)
        nx.write_graphml(G, filename)
        #nx.write_gml(G, filename)
            
    def _get_pos(self, G):
        numenv = len(self.trimmed[0].flatten())
        numnodes = numenv + len(self.trimmed)

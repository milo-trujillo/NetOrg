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
plt.ion()

from results import Results

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
        self.first_layer = 2 # How many agents in the first mid-layer
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
            if( i < self.first_layer ):
                a.create_in_vec(self.num_environment)
                a.create_state_matrix(self.num_environment)
                a.create_out_matrix(self.num_environment)
            # Second wave and up
            else:
                #old_version = created.pop(0)
                #a.set_predecessor(old_version)
                a.create_in_vec(self.num_environment + self.first_layer)
                a.create_state_matrix(self.num_environment + self.first_layer)
                a.create_out_matrix(self.num_environment + self.first_layer)

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
            if( a.num < self.first_layer ):
                indata = inenv
                innoise = envnoise
            # Second wave+
            else:
                '''
                We're only listening to the first wave nodes (no environment),
                so can skip all the inenv and envnoise steps
                '''
                loadLayerStart = 0
                loadLayerEnd = self.first_layer
                msgdata = tf.concat(self.outputs[loadLayerStart:loadLayerEnd], 1)
                indata = tf.concat([inenv, msgdata], 1)
                commnoise = tf.random_normal([self.batchsize, self.first_layer], stddev=a.noiseinstd, dtype=tf.float64)
                innoise = tf.concat([envnoise, commnoise], 1)

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

    def listening_cost(self, exponent=2):
        summed = [x.listen_cost(exponent) for x in self.agents]
        totalc = tf.add_n(summed)
        return totalc

    def speaking_cost(self, exponent=2):
        summed = [tf.reduce_sum(tf.abs(x.out_weights))**exponent for x in self.agents]
        totalc = tf.add_n(summed)
        return totalc

    # Gets the difference^2 of how far each agent is from real avg of variables
    # Note: We only look at the upper layer (A0_1, not A0_0) for determining welfare
    def loss(self, exponent=2):
        realValue = tf.reduce_mean(self.environment, 1, keep_dims=True)
        differences = [tf.reduce_mean((realValue - a.state)**exponent) for a in self.agents[self.first_layer:]]
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
        return Results(training_res, listen_params, self.num_agents, self.num_environment, welfare, self.first_layer)
    

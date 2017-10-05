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
    def __init__(self, num_environment, num_agents, num_managers, innoise,
                     outnoise, fanout, statedim, envnoise, envobsnoise,
                     batchsize, optimizer, layers, randomSeed=False, tensorboard=None, **kwargs):
        if( randomSeed == False ):
            tf.set_random_seed(634)
        self.num_environment = num_environment
        self.num_agents = num_agents
        self.num_managers = num_managers
        self.batchsize = batchsize
        self.envobsnoise = envobsnoise
        self.layers = layers
        self.agents = []
        for i in range(num_agents * self.layers):
            self.agents.append(Agent(innoise, outnoise, i, fanout, statedim, batchsize, num_agents, num_environment))
        self.environment = tf.random_normal([self.batchsize, num_environment], mean=0.0, stddev=1.0, dtype=tf.float64)
        zero = tf.convert_to_tensor(0.0, tf.float64)
        greater = tf.greater(self.environment, zero)
        self.environment = tf.where(greater, tf.ones_like(self.environment), tf.zeros_like(self.environment))
        self.build_org()
        self.objective = self.loss()
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
        indim = self.num_environment
        for i, a in enumerate(self.agents):
            created.append(a)
            # First wave
            a.create_in_vec(indim)
            a.create_state_matrix(indim)
            a.create_out_matrix(indim)
            indim += a.fanout
            # There is only one wave in this model

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
            indata = inenv
            for msg in self.outputs:
                indata = tf.concat([indata, msg], 1)
            envnoise = envnoise
            #commnoise = tf.random_normal([self.batchsize, self.num_agents], stddev=a.noiseinstd, dtype=tf.float64)
            commnoise = tf.random_normal([self.batchsize, a.indim - self.num_environment], stddev=a.noiseinstd, dtype=tf.float64)
            innoise = tf.concat([envnoise, commnoise], 1)

            # Add noise inversely-proportional to listening strength
            noisyin = indata + innoise/a.listen_weights

            # Since listen weights is 1xin we get row wise division.
            if( a.predecessor != None ):
                noisyin = tf.concat([a.predecessor.received_messages, noisyin], 1)
            a.set_received_messages(noisyin)

            # Let's pin state to either 0 or 1, for "pattern or no"
            state = tf.tanh(tf.matmul(noisyin, a.state_weights))
            zero = tf.convert_to_tensor(0.0, tf.float64)
            state = tf.where(tf.greater(state, zero), tf.ones_like(state), tf.zeros_like(state))

            a.state = state
            self.states.append(state)

            outnoise = tf.random_normal([self.batchsize, a.fanout], stddev=a.noiseoutstd, dtype=tf.float64)
            prenoise = tf.matmul(noisyin, a.out_weights)

            # Similarly, we'll pin our output message to either zero or one
            output = tf.tanh(prenoise + outnoise)
            output = tf.where(tf.greater(output, zero), tf.ones_like(output), tf.zeros_like(output))

            self.outputs.append(output)
            # output is a vector with dimensions [1, batchsize]
            #with tf.Session() as sess:
                #init = tf.global_variables_initializer()
                #sess.run(init)
                #res = sess.run(output)
                #print "Appending output for agent " + str(i) + ": " + str(res)

    # Barrier function for listening costs
    def listening_cost(self, steepness=1.0, barrier=3.0, offset=2.0):
        neg = tf.convert_to_tensor(-1.0, dtype=tf.float64)
        steep = tf.multiply(neg, tf.convert_to_tensor(steepness, dtype=tf.float64))
        penalty = tf.convert_to_tensor(100.0, dtype=tf.float64)
        barrier_t = tf.convert_to_tensor(barrier, dtype=tf.float64)
        offset_t = tf.convert_to_tensor(offset, dtype=tf.float64)
        summed = []
        for x in self.agents:
            weight = x.listen_cost()
            distance_from_barrier = tf.subtract(barrier_t, weight)
            border = tf.add(tf.multiply(steep, tf.log(distance_from_barrier)), offset_t)
            over = tf.subtract(weight, barrier_t)
            border = tf.where(tf.is_nan(border), tf.multiply(over, penalty), border)
            #border = tf.Print(border, [border]) # Debugging
            summed += [tf.maximum(tf.convert_to_tensor(0.0, dtype=tf.float64), border)]
        totalc = tf.add_n(summed)
        return totalc

    # Barrier function for speaking costs
    # speaking cost == 0 unless someone speaks too much, then shoots to
    # near infinity. This simulates a hard constraint on speaking
    # using only soft-constraints, which are more natural in Tensorflow.
    # Yes, I'm saying this is the *more* natural solution.
    def speaking_cost(self, steepness=1.0, barrier=3.0, offset=2.0):
        # This is disgusting looking, but comes out to:
        #     cost = (-steepness * log(barrier - sum_of_weights)) + offset
        # We pin the cost to 0+ so we never get negative costs, then we add
        # costs per agent together and return "total speaking cost for network"
        summed = []
        neg = tf.convert_to_tensor(-1.0, dtype=tf.float64)
        steep = tf.multiply(neg, tf.convert_to_tensor(steepness, dtype=tf.float64))
        barrier_t = tf.convert_to_tensor(barrier, dtype=tf.float64)
        offset_t = tf.convert_to_tensor(offset, dtype=tf.float64)
        for x in self.agents:
            speak_sum = tf.reduce_sum(tf.abs(x.out_weights))
            border = tf.add(tf.multiply(steep, tf.log(tf.subtract(barrier_t, speak_sum))), offset_t)
            summed += [tf.maximum(tf.convert_to_tensor(0.0, dtype=tf.float64), border)]
        totalc = tf.add_n(summed)
        return totalc

    # We look for a sequence of three 1s from the env in a row
    # If (sequence && agent returned 1) || (!sequence && agent returned 0)
    # then welfare improves. Else welfare worsened.
    # Note: We only look at non middle-managers for accuracy, but all for costs
    def loss(self, exponent=2):
        lastLayer = self.num_agents * (self.layers - 1)
        pattern = self.pattern_detected()
        pattern = tf.Print(pattern, [pattern], message="Pattern: ", summarize=100)
        incorrect = tf.Variable(0.0, dtype=tf.float64)
        zero = tf.convert_to_tensor(0.0, dtype=tf.float64)
        one = tf.convert_to_tensor(1.0, dtype=tf.float64)
        differences = []
        print "Loss function initialized"
        for a in self.agents[lastLayer+self.num_managers:]:
            print "Handling agent %d" % a.num
            a.state = tf.Print(a.state, [a.state], message="Agent State: ", summarize=100)
            match_yes = tf.logical_and(pattern, tf.equal(a.state, 1.0))
            match_no = tf.logical_and(tf.logical_not(pattern), tf.equal(a.state, 0.0))
            match = tf.logical_or(match_yes, match_no)
            differences.append(tf.cond(match, lambda: zero, lambda: one))
        differenceSum = tf.add_n(differences)
        cost = self.listening_cost() + self.speaking_cost()
        loss = differenceSum + cost
        print "Done running loss function"
        return loss

    # Implemented Justin's matrix pattern detection
    # It's real nifty!
    def pattern_detected(self):
        pattern_length = 3
        rows = self.num_environment
        cols = self.num_environment - (pattern_length - 1)
        A = np.zeros(shape=(rows, cols))
        for i in range(cols):
            A[i:(i+pattern_length), i] = 1
        # We want Y = np.dot(self.environment, A)
        # but Tensorflow doesn't have a dotproduct operatior
        Y = tf.tensordot(self.environment, A, 1)
        patterns = []
        for r in range(self.batchsize):
            rowsum = tf.reduce_max(Y[r])
            patterns += [tf.greater_equal(rowsum, pattern_length)]
        pattern = tf.stack(patterns)
        #pattern = tf.greater_equal(tf.reduce_max(Y), pattern_length)
        return pattern

    '''
    # Returns a [batchsize, 1] tensor that yields true if env at that batch
    # has three 1s in a row, yields false otherwise
    def pattern_detected(self):
        one = tf.convert_to_tensor(1.0, tf.float64)
        true = tf.convert_to_tensor(True, tf.bool)
        false = tf.convert_to_tensor(False, tf.bool)
        pattern = tf.zeros_like(self.environment[:, 0])
        for i in range(0, self.num_environment - 2):
            v1 = self.environment[:, i+0]
            v2 = self.environment[:, i+1]
            v3 = self.environment[:, i+2]
            val1 = tf.equal(v1, one)
            val2 = tf.equal(v2, one)
            val3 = tf.equal(v3, one)
            p = tf.logical_and(tf.logical_and(val1, val2), val3)
            isPattern = tf.equal(p, true)
            pattern = tf.cond(isPattern, lambda: true, lambda: pattern)
        return pattern
    '''

    # For helping graph welfare
    def welfareDifference(self, exponent=2):
        lastLayer = self.num_agents * (self.layers - 1)
        pattern = self.pattern_detected()
        incorrect = tf.Variable(0.0, dtype=tf.float32)
        zero = tf.convert_to_tensor(0.0, dtype=tf.float32)
        one = tf.convert_to_tensor(1.0, dtype=tf.float32)
        differences = []
        for a in self.agents[lastLayer+self.num_managers:]:
            match_yes = tf.logical_and(pattern, tf.equal(a.state, 1.0))
            match_no = tf.logical_and(tf.logical_not(pattern), tf.equal(a.state, 0.0))
            match = tf.logical_or(match_yes, match_no)
            differences += [tf.cond(match, lambda: zero, lambda: one)]
        differenceSum = tf.add_n(differences)
        return differenceSum

    # For helping graph welfare
    def welfareCost(self, exponent=2):
        return self.listening_cost() + self.speaking_cost()

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
        welfareDiffGen = self.welfareDifference()
        welfareCostGen = self.welfareCost()
        welfareDiff = self.sess.run(welfareDiffGen)
        welfareCost = self.sess.run(welfareCostGen)
        if( verbose ):
            print "Listen_params now set to: " + str(listen_params)
        if( self.writer != None ):
            self.writer.close()
        return Results(training_res, listen_params, self.num_agents, self.num_environment, welfare, welfareDiff, welfareCost)
    

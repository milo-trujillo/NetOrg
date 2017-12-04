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
        # Env centered on zero w/ 1 std-dev
        self.environment = tf.random_normal([self.batchsize, num_environment], mean=0.0, stddev=1.0, dtype=tf.float64)
        # Env now more likely to be zero as env size increases
        #self.environment = tf.random_uniform([self.batchsize, num_environment], minval=-1 * (1.1 ** self.num_environment), maxval=1, dtype=tf.float64)
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
            a.create_out_matrix(indim + 1) # Plus one for bias
            indim += a.fanout
            # There is only one wave in this model

    def build_wave(self):
        """
        This loops through agents building the tensorflow objects
        that determine the agents states and outputs that are then
        recursively use to build all other agent's states and outputs
        """
        self.outputs = []
        threshold = tf.convert_to_tensor(0.5, tf.float64)
        lastLayer = self.num_agents * (self.layers - 1)
        for i, a in enumerate(self.agents):
            envnoise = tf.random_normal([self.batchsize, self.num_environment], stddev=self.envobsnoise, dtype=tf.float64)
            inenv = self.environment

            incomm = None

            # First wave
            indata = inenv
            for msg in self.outputs:
                indata = tf.concat([indata, msg], 1)

            # And add the bias, which has no noise
            biasedin = tf.concat([tf.constant(1.0, dtype=tf.float64, shape=[self.batchsize, 1]), indata], 1)

            # Since listen weights is 1xin we get row wise division.
            if( a.predecessor != None ):
                biasedin = tf.concat([a.predecessor.received_messages, biasedin], 1)

            a.set_received_messages(biasedin)
            output = tf.sigmoid(tf.matmul(biasedin, a.out_weights))
			a.action = output

            self.outputs.append(output)


    # Implemented Wolpert's model for Dunbars number
    def dunbar_listening_cost(self, dunbar=3):
        penalties = []
        for x in self.agents:
            weights = tf.Print(x.listen_weights, [x.listen_weights], message="Listen weight: ")
            top_k = tf.transpose(tf.nn.top_k(weights, dunbar+1).values)
            top_k = tf.Print(top_k, [top_k], message="Top k: ", summarize=100)
            top = tf.log(top_k[0])
            bottom = tf.log(top_k[dunbar])
            cost = tf.sigmoid(tf.subtract(top, bottom))
            penalties += [cost]
        penalty = tf.stack(penalties)
        return tf.reduce_prod(penalty)

    # Justin's model for Dunbar's number
    def dunbar_listening_barrier(self, dunbar=3, harshness=100, steepness=0.1):
        penalties = []
        neg = tf.convert_to_tensor(-1.0, dtype=tf.float64)
        zero = tf.convert_to_tensor(0, dtype=tf.float64)
        for x in self.agents:
            count = tf.reduce_sum(tf.abs(tf.tanh(tf.multiply(harshness, x.listen_weights))))
            msg = "Agent " + str(x.num) + " has a non-zero count of: "
            count = tf.Print(count, [count], message=msg)
            border = tf.multiply(steepness, tf.log(tf.subtract(dunbar, count)))
            penalties += [border]
            #penalties += [tf.maximum(zero, border)]
        penalty = tf.reduce_sum(tf.stack(penalties))
        return penalty

    # We look for a sequence of three 1s from the env in a row
    # If (sequence && agent returned 1) || (!sequence && agent returned 0)
    # then welfare improves. Else welfare worsened.
    # Note: We only look at non middle-managers for accuracy, but all for costs
    def loss(self, exponent=2):
        lastLayer = self.num_agents * (self.layers - 1)
        pattern = self.pattern_detected()
        #pattern = tf.Print(pattern, [pattern], message="Pattern: ", summarize=100)
        incorrect = tf.Variable(0.0, dtype=tf.float64)
        zero = tf.convert_to_tensor(0.0, dtype=tf.float64)
        one = tf.convert_to_tensor(1.0, dtype=tf.float64)
        punishments = []
        print "Loss function initialized"
        for a in self.agents[lastLayer+self.num_managers:]:
            state = tf.reshape(a.action, [-1]) # Flatten array
            #state = tf.Print(state, [state], message="Agent State: ", summarize=100)
            punishments.append(self.agent_punishment(pattern, state))
        punishmentSum = tf.divide(tf.add_n(punishments), self.batchsize)
        cost = self.dunbar_listening_cost()
        loss = punishmentSum + cost
        print "Done running loss function"
        return loss

    # This is a reward function to be run for each agent
    # Since we've phrased the problem as minimization
    # rather than maximization, it's technically a punishment
    def agent_punishment(self, pattern, state):
        eps = tf.convert_to_tensor(0.0001, dtype=tf.float64)
        neg = tf.convert_to_tensor(-1.0, dtype=tf.float64)
        one = tf.convert_to_tensor(1.0, dtype=tf.float64)
        one_minus_pattern = tf.subtract(one, pattern)
        one_minus_state = tf.subtract(one, state)
        one_minus_state_plus_eps = tf.add(eps, one_minus_state)
        one_plus_pattern = tf.add(one, pattern)
        eps_plus_state = tf.add(eps, state)
        yes_pattern = tf.multiply(pattern, tf.log(eps_plus_state))
        #yes_pattern = tf.Print(yes_pattern, [yes_pattern], message="Yes Pattern: ", summarize=100)
        no_pattern = tf.multiply(one_minus_pattern, tf.log(one_minus_state_plus_eps))
        #no_pattern = tf.Print(no_pattern, [no_pattern], message="No Pattern: ", summarize=100)
        punishment = tf.multiply(neg, tf.add(yes_pattern, no_pattern))
        return tf.reduce_sum(punishment)

    '''
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
            patterns += [tf.cast(tf.greater_equal(rowsum, pattern_length), tf.float64)]
        pattern = tf.stack(patterns)
        #pattern = tf.greater_equal(tf.reduce_max(Y), pattern_length)
        return pattern
    '''

    '''
    # Are more than half the environment nodes set to 1?
    def pattern_detected(self):
        patterns = []
        for r in range(self.batchsize):
            rowsum = tf.reduce_max(self.environment[r])
            patterns += [tf.cast(tf.greater_equal(rowsum, self.num_environment / 2), tf.float64)]
        return tf.stack(patterns)
    '''

    # Do the left and right sides of env both have an even/odd number of 1s?
    def pattern_detected(self):
        patterns = []
        midpoint = self.num_environment/2
        left = tf.slice(self.environment, [0, 0], [self.batchsize, midpoint])
        right = tf.slice(self.environment, [0, midpoint], [self.batchsize, self.num_environment - midpoint])
        leftsum = tf.reduce_sum(left, 1)
        #leftsum = tf.Print(leftsum, [leftsum], message="leftsum: ")
        rightsum = tf.reduce_sum(right, 1)
        #rightsum = tf.Print(rightsum, [rightsum], message="rightsum: ")
        lmod = tf.mod(leftsum, 2)
        #lmod = tf.Print(lmod, [lmod], message="Lmod: ")
        rmod = tf.mod(rightsum, 2)
        #rmod = tf.Print(rmod, [rmod], message="Rmod: ")
        pattern = tf.cast(tf.equal(lmod, rmod), tf.float64)
        return pattern

    # For helping graph welfare
    # This should be exactly the same as 'loss', except without
    # any communication costs or debugging statements
    def welfareDifference(self):
        lastLayer = self.num_agents * (self.layers - 1)
        pattern = self.pattern_detected()
        incorrect = tf.Variable(0.0, dtype=tf.float64)
        zero = tf.convert_to_tensor(0.0, dtype=tf.float64)
        one = tf.convert_to_tensor(1.0, dtype=tf.float64)
        punishments = []
        for a in self.agents[lastLayer+self.num_managers:]:
            state = tf.reshape(a.action, [-1]) # Flatten array
            punishments.append(self.agent_punishment(pattern, state))
        punishmentSum = tf.divide(tf.add_n(punishments), self.batchsize)
        loss = punishmentSum
        return loss

    # For helping graph welfare
    def welfareCost(self, exponent=2):
        return self.dunbar_listening_cost()

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
                lastLayer = self.num_agents * (self.layers - 1)
                for a in self.agents[lastLayer+self.num_managers:]:
                    a.listen_weights = tf.Print(a.listen_weights, [a.listen_weights], message="Listen weights: ")
                #listen_params = self.sess.run([a.listen_weights for a in self.agents])
                #output_params = self.sess.run([a.out_weights for a in self.agents])
                #print "Listen_params now set to: " + str(listen_params)
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
    

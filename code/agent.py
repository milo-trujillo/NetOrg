#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from itertools import count

class Agent(object):
    """
    An agent in an organization

    Note to self: tf.get_variable will *create* variables if they don't exist already!
    
    """
    _ids = count(0)
    def __init__(self, noiseinstd, noiseoutstd, num, fanout, statedim, batchsize, numagents, numenv, prevState=None, **kwargs):
        self.id = next(self._ids)
        self.num = num
        self.statedim = statedim
        self.fanout = fanout
        self.noiseinstd = noiseinstd
        self.noiseoutstd = noiseoutstd
        self.batchsize = batchsize
        self.numagents = numagents
        self.numenv = numenv
        self.prevState = prevState
        self.predecessor = None
        self.received_messages = None

    def create_in_vec(self, indim):
        print "Creating in vec for agent %d" % self.num
        self.indim = indim
        # If A0_1 is agent 15 and there's 10 agents, then A0_0 will be agent 5, but will be LISTEN_WEIGHT[prev_gen + num_env]
        prev_gen = self.num - self.numagents
        prev_listen = prev_gen + self.numenv
        self.listen_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "listen" +str(self.id) , shape=[1, indim])
        #with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            #print "Agent %d Created with listen_weights: %s" % (self.num, str(sess.run(self.listen_weights)))

    def get_listen_weights(self, env_exponent_cost):
        if( self.predecessor == None ):
            return tf.reduce_sum(tf.abs(self.listen_weights))**env_exponent_cost
        else:
            # We want to assign a penalty to talking to people with different objectives
            # here we've made the cost 2x
            multiplier = np.empty_like(self.listen_weights)
            if( self.num % 2 == 0 ):
                multiplier[0::2] = 1.0
                multiplier[1::2] = 2.0
            multiplied_weight = tf.matmul(multiplier, tf.abs(self.listen_weights), transpose_a=True)
            return tf.add(tf.reduce_sum(multiplied_weight, self.predecessor.get_listen_weights(env_exponent_cost)))

    def get_out_weights(self):
        if( self.predecessor == None ):
            return tf.reduce_sum(tf.abs(self.out_weights))
        else:
            return tf.add(tf.reduce_sum(tf.abs(self.out_weights)), self.predecessor.get_out_weights())

    def listen_cost(self, exponent, env_exponent_cost):
        if( self.predecessor == None ):
            return (tf.reduce_sum(tf.abs(self.listen_weights))**env_exponent_cost)**exponent
        else:
            return tf.add(tf.reduce_sum(tf.abs(self.listen_weights)), self.predecessor.get_listen_weights(env_exponent_cost))**exponent

    def speaking_cost(self, exponent):
        if( self.predecessor == None ):
        	return tf.reduce_sum(tf.abs(self.out_weights))**exponent
        else:
            return tf.add(tf.reduce_sum(tf.abs(self.out_weights)), self.predecessor.get_out_weights())**exponent

    def create_state_matrix(self, indim):
        self.indim = indim
        self.state_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "state" +str(self.id), shape=[indim, self.statedim])

    def set_predecessor(self, agent):
        self.predecessor = agent

    def set_received_messages(self, msgs):
        self.received_messages = msgs

    def create_out_matrix(self, indim):
        self.indim = indim
        self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "out" +str(self.id), shape=[indim, self.fanout])
        '''
        Indim is the number of independent messages we can send. We can send one
        '''
        #with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            #print "Agent %d Created with out_weights: %s" % (self.num, str(sess.run(self.out_weights)))

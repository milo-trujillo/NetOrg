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
    def __init__(self, noiseinstd, noiseoutstd, num, fanout, statedim, batchsize, numagents, numenv, **kwargs):
        self.id = next(self._ids)
        self.num = num
        self.statedim = statedim
        self.fanout = fanout
        self.noiseinstd = noiseinstd
        self.noiseoutstd = noiseoutstd
        self.batchsize = batchsize
        self.numagents = numagents
        self.numenv = numenv
        self.predecessor = None
        self.received_messages = [None * (self.num_agents + 1)]

    def create_in_vec(self, indim):
        self.indim = indim
        #self.listen_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "listen" +str(self.id) , shape=[1, self.numagents])
        self.listen_weights = []
        for i in range(0, self.numagents + 1):
            if( i == 0 ):
                n = str(self.num) + "listen" + str(self.id) + str(i)
                self.listen_weights.append(tf.get_variable(dtype=tf.float64, name=n, shape=[1, indim]))
            #elif( self.num == i + 1 ):
                #self.listen_weights.append(tf.Variable(tf.zeros_like(self.listen_weights[0]), trainable=False))
            else:
                self.listen_weights.append(tf.identity(self.listen_weights[0]))
        #with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            #print "Agent %d Created with listen_weights: %s" % (self.num, str(sess.run(self.listen_weights[0])))

    def create_state_matrix(self, indim):
        #self.state_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "state" +str(self.id), shape=[self.numagents, self.statedim])
        self.state_weights = []
        for i in range(0, self.numagents + 1):
            if( i == 0 ):
                self.state_weights.append(tf.get_variable(dtype=tf.float64, name=str(self.num) + "state" + str(self.id) + str(i), shape=[indim, self.statedim]))
            else:
                self.state_weights.append(tf.Variable(self.state_weights[0].initialized_value()))
        #with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            #print "Agent %d Created with state matrix: %s" % (self.num, str(sess.run(self.state_weights[0])))

    def set_predecessor(self, agent):
        self.predecessor = agent

    def set_received_messages(self, iteration, msgs):
        self.received_messages[i] = msgs

    def create_out_matrix(self, indim):
        #self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "out" +str(self.id), shape=[self.numagents, self.fanout])
        self.out_weights = []
        for i in range(0, self.numagents + 1):
            if( i == 0 ):
                self.out_weights.append(tf.get_variable(dtype=tf.float64, name=str(self.num) + "out" + str(self.id) + str(i), shape=[indim, self.fanout]))
            elif( i == (self.num + 1) ):
                self.out_weights.append(tf.Variable(tf.zeros_like(self.out_weights[0]), trainable=False))
            else:
                self.out_weights.append(tf.Variable(self.out_weights[0].initialized_value()))
        #with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            #print "Agent %d Created with out matrix: %s" % (self.num, str(sess.run(self.out_weights[0])))

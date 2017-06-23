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
    def __init__(self, noiseinstd, noiseoutstd, num, fanout, statedim, batchsize, numagents, **kwargs):
        self.id = next(self._ids)
        self.num = num
        self.statedim = statedim
        self.fanout = fanout
        self.noiseinstd = noiseinstd
        self.noiseoutstd = noiseoutstd
        self.batchsize = batchsize
        self.numagents = numagents

    def create_in_vec(self, indim):
        self.indim = indim
        #self.listen_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "listen" +str(self.id) , shape=[1, self.numagents])
        self.listen_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "listen" +str(self.id) , shape=[1, indim])
        #with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            #print "Agent %d Created with listen_weights: %s" % (self.num, str(sess.run(self.listen_weights)))

    def create_state_matrix(self, indim):
        #self.state_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "state" +str(self.id), shape=[self.numagents, self.statedim])
        self.state_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "state" +str(self.id), shape=[indim, self.statedim])

    def create_out_matrix(self, indim):
        #self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "out" +str(self.id), shape=[self.numagents, self.fanout])
        self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "out" +str(self.id), shape=[indim, self.fanout])

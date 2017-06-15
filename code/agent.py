#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from itertools import count



class Agent(object):
    """
    An agent in an organization
    
    """
    _ids = count(0)
    def __init__(self, noiseinstd, noiseoutstd, num, fanout, statedim, batchsize, **kwargs):
        self.id = next(self._ids)
        self.num= num
        self.statedim = statedim
        self.fanout = fanout
        self.noiseinstd = noiseinstd
        self.noiseoutstd = noiseoutstd
        self.batchsize = batchsize

    def create_in_vec(self, indim):
        # Activation Functions
        self.indim = indim
        #init = tf.constant(np.random.rand(1, indim))
        self.listen_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "listen" +str(self.id) , shape=[1, indim])
        #self.listen_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "listen" ,  initializer=init)

    def create_state_matrix(self, indim):
        # Activation Functions
        #init = tf.constant(np.random.rand(indim, self.statedim))
        self.state_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "state" +str(self.id), shape=[indim, self.statedim])
        #self.state_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "state" , initializer=init)

    def create_out_matrix(self, indim):
        # Activation Functions
        #init = tf.constant(np.random.rand(indim, self.fanout))
        self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "out" +str(self.id), shape=[indim, self.fanout])
        #self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "out", initializer=init)

    def random_reset(self):
        listen = np.random.normal(size=(1, self.indim))
        state = np.random.normal(size= (self.indim, self.statedim))
        out = np.random.normal(size=(self.indim, self.fanout))
        assignments = [tf.assign(self.listen_weights, listen), tf.assign(self.state_weights, state), tf.assign(self.out_weights, out)]
        return assignments
    
        
    

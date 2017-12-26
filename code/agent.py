#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from itertools import count

class Agent(object):
    """
    An agent in an organization

    Note to self: tf.get_variable will *create* variables if they don't exist already!

    In the nonlinear model agents are just a wrapper around one vector,
    their "out weights". This vector is responsible for determining what they
    listen to and how they translate those observations in to outgoing messages
    
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
        self.received_messages = None

    # This is for multi-layer systems.
    # In a multi-layer system, we say an agent inherits the observations of its
    # predecessor, so you can make decisions based on your observations at any
    # previous timestep. It is currently unused in nonlinear systems.
    def set_received_messages(self, msgs):
        self.received_messages = msgs

    # Run once at startup in network.py to initialize the out weights and 
    # (optionally) print debugging data
    # Note: Indim is how many Independent Messages each agent can send
    # In other words, if indim=2 then each agent can send two distinct
    # messages based on their observations. This is currently nonsensical
    # in the nonlinear model.
    def create_out_matrix(self, indim):
        self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "out" +str(self.id), shape=[indim, self.fanout])
        #with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            #print "Agent %d Created with out_weights: %s" % (self.num, str(sess.run(self.out_weights)))

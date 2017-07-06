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

    def create_in_vec(self, indim):
        print "Creating in vec for agent %d" % self.num
        self.indim = indim
        # If A0_1 is agent 15 and there's 10 agents, then A0_0 will be agent 5, but will be LISTEN_WEIGHT[prev_gen + num_env]
        prev_gen = self.num - self.numagents
        prev_listen = prev_gen + self.numenv
        orig_listen_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "listen" +str(self.id) , shape=[1, indim])
        # If there was a previous version of this node, we want to listen to it perfectly (at no cost)
        # See: https://stackoverflow.com/questions/39859516/how-to-update-a-subset-of-2d-tensor-in-tensorflow
        if( prev_gen >= 0 and indim >= prev_listen ):
            self.had_previous = True
            row = tf.gather(orig_listen_weights, 0)
            newrow = tf.concat([row[:prev_listen], tf.constant([9999999999999.9], dtype=tf.float64), row[prev_listen+1:]], axis=0)
            self.listen_weights = tf.reshape(newrow, [1, indim])
            #self.listen_weights = tf.scatter_update(orig_listen_weights, tf.constant(0), newrow)
        else:
            self.had_previous = False
            self.listen_weights = orig_listen_weights
        #with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            #print "Agent %d Created with listen_weights: %s" % (self.num, str(sess.run(self.listen_weights)))

    def listen_cost(self, exponent):
        if( self.had_previous ):
            print "Calculating limited cost for agent %d" % self.num
            prev_gen = self.num - self.numagents
            prev_listen = prev_gen + self.numenv
            row = tf.gather(self.listen_weights, 0)
            countable_weight = tf.concat([row[:prev_listen], row[prev_listen+1:]], axis=0)
            return tf.reduce_sum(tf.abs(countable_weight))**exponent
        else:
            print "Calculating unlimited cost for agent %d" % self.num
            return tf.reduce_sum(tf.abs(self.listen_weights))**exponent

    def create_state_matrix(self, indim):
        self.state_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "state" +str(self.id), shape=[indim, self.statedim])

    def create_out_matrix(self, indim):
        self.out_weights = tf.get_variable(dtype=tf.float64, name=str(self.num) + "out" +str(self.id), shape=[indim, self.fanout])
        '''
        Indim is the number of independent messages we can send. We can send one
        '''
        #with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            #print "Agent %d Created with out_weights: %s" % (self.num, str(sess.run(self.out_weights)))

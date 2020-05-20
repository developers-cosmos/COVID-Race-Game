#!/usr/bin/env python

## @package bayse_filter.py
#
# Massimiliano Patacchiola, Plymouth University 2016
#
# Discrete Bayes filter (DBF) implementation. It permits estimating
# the value of a quantity X given the observation Z.
# If we have some noisy measurement Z of a discrete quantity X, we  
# can use the DBF  in a recursive estimationto find the most 
# probable value of X (belief). 


import numpy as np
import sys


class DiscreteBayesFilter:

    def __init__(self, states_number):
        if(states_number<=0):
            raise ValueError('BayesFilter: the number of states must be greater than zero.')
        else: 
            self._states_number = states_number
            #Declaring the prior distribution
            self._prior = np.zeros(states_number, dtype=np.float32)
            self._prior.fill(1.0/states_number) #uniform
            #Declaring the poterior distribution
            #self._posterior = np.empty(states_number, dtype=np.float32)
            #self._posterior.fill(1.0/states_number) #uniform
            #Declaring the conditional probability table
            self._cpt = np.zeros((states_number, states_number), dtype=np.float32)
            self._cpt.fill(1.0/states_number) #uniform
            #Declaring the evidence scalar
            #self._evidence = -1 #negative means undefined

    ##
    # Return the posterior probability given a prior and an evidence.
    # @param prior is a 1 dimensional array, it represents the distribution of X at t_0
    # @param cpt is a matrix, it is the conditional probability table of Z|X
    def initialise(self, prior, cpt):
        if(prior.shape[0]!=self._states_number):
            raise ValueError('DiscreteBayesFilter: the shape of the prior is different from the total number of states.')
        elif(cpt.shape[0]!=self._states_number or cpt.shape[1]!=self._states_number):
            raise ValueError('DiscreteBayesFilter: the shape of the cpt is different from the total number of states.')
        else: 
            self._prior = prior.copy()
            self._cpt = cpt.copy()


    ##
    # After the initialisation it is possible to predict the current value
    # of the quanity X. It is necessary to pass the cpt_motion_model and the
    # belief.
    #
    # @param cpt_motion_model is the conditional probability table of
    # x(t)|x(t-1). The control u is not used in our case
    # @param belief is the probability distribution vector associated with x
    # @return it returns the posterior distribution of X given the evidence
    def predict(self, belief, cpt_motion_model):
        belief = np.matrix(belief) * cpt_motion_model
        return np.asarray(belief).reshape(-1)


    def update(self, belief_predicted, measure, cpt_measure_accuracy):
        #Getting P(Z) and Likelihood
        p_z = np.sum(cpt_measure_accuracy[:,measure]) #scalar P(Z)
        likelihood = cpt_measure_accuracy[:,measure].copy() #vector P(Z|X)
        #likelihood = np.transpose(likelihood)
        #Getting the posterior distribution
        belief_updated = np.multiply(belief_predicted, likelihood)
        belief_updated /= p_z #vector P(X|Z)
        #Normalise to sum up to 1
        normalisation = np.sum(belief_updated)
        belief_updated /= normalisation
        return belief_updated

    

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:16:12 2014

@author: bao
"""

import numpy as np
from sys import stdout

def transition_probability(energy, energy_new, T):
    """Returns the transition probability given the energy of two
    states and the current temperature.
    """
    if energy_new < energy:
        return 1.0
    else:
        return np.exp(-(energy_new - energy) / T)
        

def temperature_cauchy(k, T0):
    """Cauchy schedule to update tempertature at step k.
    """
    T_new = T0 / (1 + k)
    return T_new


def temperature_boltzmann(k, T0):
    """Boltzmann schedule to update tempertature at step k.
    """
    T_new = T0 / np.log(1 + k+1)
    return T_new


def anneal(initial_state, energy_function, neighbour, transition_probability, temperature, max_steps, energy_max, T0, log_every=1000):
    """Simulated annealing optimization.

    initial_state
    energy_function
    neighbour
    transition_probability
    temperature
    max_steps
    energy_max
    """
    state = initial_state
    energy = energy_function(state)
    state_best = state
    energy_best = energy
    energy_old = energy
    k = 0
    iteration_step = k
    #print "Step) Energy \t Prob. \t Temp. \t (E'-E) \t BEST"
    while k < max_steps and energy > energy_max:
        T = temperature(k, T0)
        state_new = neighbour(state)
        energy_new = energy_function(state_new)
        p = transition_probability(energy, energy_new, T)
        if p > np.random.rand():
            state = state_new
            energy_old = energy
            energy = energy_new
            
        if energy_new < energy_best:
            state_best = state_new
            energy_best = energy_new
            iteration_step = k
            #print "* %s) %s \t %s \t %s \t %s \t %s" % (k, energy_best, p, T, energy_old, energy_new)
            

        if (k % log_every) == 0:
            #print "%s) %s \t %s \t %s \t %s \t %s" % (k, energy, p, T, energy_new - energy_old, energy_best)
            stdout.flush()
            
        k += 1
    print "The minimize of energy:  ", energy_best, "  at iteration : ", iteration_step
    return state_best, energy_best
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:54:37 2014

@author: bao


 mapping prototype of tract1 to find the coressponding in tract2
 using the new loss function based on probability
 
 result is the probability of each mapp 
 
 input: tract1(size n), and tract2(size m)
 output: the probability mapping matrix (n,m)
         where pr(i,j) is the probability of streamline i in tract1 mapped to streamline j in tract2
         for each row, sum(pr(i,j)) (with all j) should be equal to 1
"""


def loss_function_Eman(dm1, dm2, pr_map12):
        """
        Computes the loss function of a given probability mapping.
        dm1, and dm2 are the distance matrices of tract1 and tract2 repestively        
        """
        
        loss = np.linalg.norm(dm1[np.triu_indices(size1)] - dm2[mapping12[:,None], mapping12][np.triu_indices(size1)])
        return loss
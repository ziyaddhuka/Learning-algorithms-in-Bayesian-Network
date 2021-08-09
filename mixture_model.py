#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 22:01:17 2021

@author: zad
"""

import networkx as nx
import pandas as pd
import numpy as np
import itertools
import sys
import time
from tqdm import tqdm
from copy import deepcopy
from scipy.special import logsumexp
from helper_functions import *


"""
function to create random DAG's
used to create cycles so not used
"""
def get_random_DAG(max_parents,graph):
    new_graph = graph.copy()
    nodes_array = np.array(new_graph.nodes)
    new_factors = []
    nodes_processed = []
    total_nodes = len(new_graph.nodes())
    initial_dict = {}
    for i in range(len(new_graph.nodes())):
        selected_child = nodes_array[i]
        ## randomly selecting 1 to 3 parents for the selected node/variable
        number_of_parents = np.random.randint(1,max_parents+1)
        ## choosing the parents from the variable array
        parents = np.random.choice(np.setdiff1d(nodes_array,selected_child),size=number_of_parents,replace=False)

        factor = np.append(parents,selected_child)
        cliques = list(map(int,factor))
        new_graph.add_edges_from(list(set(itertools.combinations(cliques, 2))))

        new_factors.append(factor)
    return new_factors


"""
function to create random DAG's without cycle
"""

def get_new_random_DAG(max_parents,graph):
    nodes_array = np.array(graph.nodes)
    nodes_completed = []
    new_factors = []
    i=0
    while nodes_array.size>0:
        # randomly selecting one child from list of nodes
        factors = []
        selected_child = np.random.choice(nodes_array)
    #     nodes_array = np.delete(nodes_array,selected_child)
        nodes_array = np.setdiff1d(nodes_array,selected_child)
        if i==0:
            number_of_parents = np.random.randint(1,max_parents+1)
            parents = np.random.choice(nodes_array,size=number_of_parents,replace=False)

            nodes_completed+=list(parents)
            # nodes_array = np.delete(nodes_array,list(parents))
            nodes_array = np.setdiff1d(nodes_array,parents)
            nodes_completed.append(selected_child)
            # factors = list(parents.reshape(-1,1))
            ## doing so will always put child on the end of the list/factor
            factor = np.append(parents,selected_child)
            for parent in parents:
                new_factors.append(np.array([parent]))
                nodes_array = np.setdiff1d(nodes_array,parent)
            new_factors.append(factor)
            i=1
        else:
            ## randomly selecting 1 to 3 parents for the selected node/variable
            max_legal_selection = min(max_parents,len(nodes_completed))
            number_of_parents = np.random.randint(1,max_legal_selection+1)
            ## choosing the parents from the variable array
            parents = np.random.choice(nodes_completed,size=number_of_parents,replace=False)
            factor = np.append(parents,selected_child)
            new_factors.append(factor)
            nodes_completed.append(selected_child)
    return new_factors


"""
function to find likelihood
"""

def find_likelihood(test_data,prob_dict_values_cp,factors,latent_pr):
    ## if it is true distribution then there won't be any latent variable so keeping it as 1
    if latent_pr == -1:
        latent_pr = 1
    mul_arr = np.ones(test_data.shape)
    for factor in factors:
        variable = factor[-1]
        assignments = test_data[:,factor]
        binary_scale = 2**np.arange(assignments.shape[1])[::-1]
        int_from_bin = assignments.dot(binary_scale)

        prob = np.take(prob_dict_values_cp[variable],int_from_bin)

        mul_arr[:,variable] = prob
    model_likelihood = np.sum(np.log10(mul_arr),axis=1)
    model_likelihood = model_likelihood + np.log10(latent_pr)
    return model_likelihood


"""
expectation step
"""
def e_step(temp_, prob_dict_values_cp, index,factors):
    mul_arr = np.ones(temp_.shape)
    for factor in factors:
        variable = factor[-1]
        assignments = temp_[:,factor]
        binary_scale = 2**np.arange(assignments.shape[1])[::-1]
        int_from_bin = assignments.dot(binary_scale)
        prob = np.take(prob_dict_values_cp[variable],int_from_bin)
        mul_arr[:,variable] = prob.copy()
    weights_t = 0
    wts = []
    for i in range(0,len(index)-1):
        tmp_mul_arr = mul_arr[index[i]:index[i+1]].copy()
        weights_t = np.prod(tmp_mul_arr,axis=1)
        weights_t = (weights_t/weights_t.sum())
        wts.append(weights_t)
    weights_t = np.concatenate(wts)
    return weights_t


"""
maximisation step
"""
def m_step(temp_ ,weights, factors,prob_dict_values):
    new_prob = prob_dict_values.copy()
    for factor in factors:
        variable = factor[-1]
        prob_arr = new_prob[variable]
        factor_assignments = temp_[:,factor].copy()
        if len(factor)<2:
            new_prob_arr = []
            idx = np.where(factor_assignments==0)[0]
            numerator = weights[idx].sum()
            denominator = weights.sum()
            pr = numerator/denominator
            new_prob_arr.append(pr)
            new_prob_arr.append(1-pr)
        else:
            binary_scale = 2**np.arange(factor_assignments.shape[1])[::-1]
            numerator_int_from_bin = factor_assignments.dot(binary_scale)
            factor_assignment_without_chid = factor_assignments[:,:-1]
    
            denominator_int_from_bin = factor_assignment_without_chid.dot(binary_scale[:-1])
            keys = list(range(0,2**len(factor)))
            numerator_count = {key:0 for key in keys}
    
            denominator_count = numerator_count.copy()
    
            unique_nos = np.unique(numerator_int_from_bin)
            for unique_ele in unique_nos:
                ix = np.where(numerator_int_from_bin==unique_ele)[0]
                numerator_count[unique_ele] = weights[ix].sum()
    
            unique_nos = np.unique(denominator_int_from_bin)
            for unique_ele in unique_nos:
                ix = np.where(denominator_int_from_bin==unique_ele)[0]
                count_sum = weights[ix].sum()
                denominator_count[unique_ele] = count_sum
                denominator_count[unique_ele+1] = count_sum
            new_prob_arr = []
            for key,value in numerator_count.items():
                if denominator_count[key]==0:
                    new_prob_arr.append(0)
                else:
                    new_prob_arr.append((value)/(denominator_count[key]))

        new_prob_arr = np.array(new_prob_arr)

        i=0
        while i<new_prob_arr.shape[0]:
#         for i in range(0,new_prob_arr.shape[0]):
#             print(new_prob_arr[i])
            if new_prob_arr[i]==0 and new_prob_arr[i+1]==0:
                new_prob_arr[i]=10**-5
                new_prob_arr[i+1] = 1-new_prob_arr[i]

            elif new_prob_arr[i]==0:
                new_prob_arr[i]=10**-5
                new_prob_arr[i+1]-=new_prob_arr[i]
            elif new_prob_arr[i+1]==0:
                new_prob_arr[i+1]=10**-5
                new_prob_arr[i]-=new_prob_arr[i+1]
            i+=2
        new_prob[variable] = new_prob_arr

    return new_prob


"""
mixture models utilizes previously created EM algorithms function in efficient manner
"""

def mixture_model(uai_file,train_file,test_data_file,k):

    probab_dict, true_factors, graph = read_model_file(uai_file)
    testing_prob_dict_values = {}
    for i in range(len(probab_dict)):
        testing_prob_dict_values[probab_dict[i].columns[-2]] = probab_dict[i]['pr'].values
    
    df = read_data(train_file)
    # df = df.dropna(axis=1)
    data_copy = df.astype(int)
    df_in_narr = np.array(data_copy)
    
    mega_lldiff = []
    for lp in range(0,5):
        print("init =", lp)
        new_graph = deepcopy(graph)
        dags_and_prob = {}
        dags_and_factors = {}
        max_parents = 3
        for i in range(0,k):
            new_factors = get_new_random_DAG(max_parents,new_graph)
            probab_dict_init = initialize_probability(new_factors)
            dags_and_prob[i] = probab_dict_init
            dags_and_factors[i] = new_factors.copy()
    
        arr_of_prob_dict = []
        for i in range(0,k):
            prob_dict_values = {}
            for j in range(len(dags_and_prob[i])):
                prob_dict_values[dags_and_prob[i][j].columns[-2]]= dags_and_prob[i][j]['pr'].values
            arr_of_prob_dict.append(prob_dict_values)
    
        latent_var_pr = []
        for i in range(0,k):
            latent_var_pr.append(np.random.uniform(0,1))
        latent_var_pr = np.array(latent_var_pr)
        latent_var_pr = latent_var_pr/latent_var_pr.sum()
    
        arr_of_prob_dict_train = arr_of_prob_dict.copy()
        latent_var_pr_train = latent_var_pr.copy()
    
        print("Starting Training")
    
    
        arr_of_prob_dict_t = arr_of_prob_dict_train.copy()
        latent_var_pr_t = latent_var_pr_train.copy()
        index = [0,df_in_narr.shape[0]]
    
        # Starting the EM Algorithm with latent variable and randomly created DAG's and its probabilities
        for i in range(0,20):
            weights = []
            print("EM Step=",i)
            # E-step
            for j in range(0,k):
                per_wt = e_step(df_in_narr,arr_of_prob_dict_t[j], index, dags_and_factors[j])
                weights.append(per_wt)
            
            # using the generated weights we multiply the corresponding weight with latent prob and normalize them 
            weight_t = np.array(weights).T
            new_weights = weight_t*latent_var_pr_t
            denominator = np.sum(new_weights,axis=1).reshape(-1,1)
            new_weights = new_weights/denominator
            # M-step
            for j in range(0,k):
                new_prob_train = m_step(df_in_narr,new_weights[:,j],dags_and_factors[j],arr_of_prob_dict_t[j])
                arr_of_prob_dict_t[j] = new_prob_train.copy()
            
            # updating latent variable
            new_lat = new_weights.sum(axis=0)
            latent_var_pr_t = new_lat/new_lat.sum()
    
        lldiff = 0
        trained_model_likelihood = []
        test_df = pd.read_csv(test_data_file,skiprows=1,header=None,delimiter=' ')
        test_df = test_df.dropna(axis=1)
        
        # finding the likelihood
        for i in range(0,k):
            sing_likel = find_likelihood(np.array(test_df),arr_of_prob_dict_t[i],dags_and_factors[i],latent_var_pr_t[i])
            trained_model_likelihood.append(sing_likel)
    
        trained_model_likelihood = np.array(trained_model_likelihood).T
        # summing the probabilities in exponent space and convert it back to logspace
        trained_model_likelihood = logsumexp(trained_model_likelihood,axis=1)
        true_model_likelihood = find_likelihood(np.array(test_df),testing_prob_dict_values,true_factors,-1)
        lldiff = np.sum(np.abs(true_model_likelihood-trained_model_likelihood))
        print('----------------------------')
        print('log likelihood difference = ', lldiff)
        print('----------------------------')
        mega_lldiff.append(lldiff)
    mega_lldiff = np.array(mega_lldiff)
    ldiff_mean = np.mean(mega_lldiff)
    ldiff_std = np.std(mega_lldiff)
    prt_str =  "{:.7f} ± {:.7f}".format(ldiff_mean, ldiff_std)
    print('----------------------------')
    print("mean and standard deviation for log likelihood sum = ", prt_str)
    print('----------------------------')
    return mega_lldiff



if __name__ == '__main__':
    model_file_path = 'hw5-data/dataset1/1.uai'
    train_file = 'hw5-data/dataset1/train-f-1.txt'
    test_data_file = 'hw5-data/dataset1/test.txt'
    k=4
    mixture_model(model_file_path,train_file,test_data_file,k)

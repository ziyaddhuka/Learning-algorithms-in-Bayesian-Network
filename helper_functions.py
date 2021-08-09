
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import sys
import time
from tqdm import tqdm
from copy import deepcopy
from scipy.special import logsumexp




def get_binary(n):
    l = []
    # loop from 0 to 2^n, used bitwise shift here
    for i in range(1<<n):
        # bin(2)='0b10' and bin(4)='0b100' but we need only the string after b
        s=bin(i)[2:]
        # expanding the binary to fit it to the dimension for e.g. 10 will be 010 in dimension = 3
        s='0'*(n-len(s))+s
        # check if the the corresponding position of evidence matches with the bit value
        # if yes then we only need those
    #     if s[var_pos]==str(evid):
        l.append(list(s))
    return(np.array(l))


def initialize_probability(factors):
    initial_prob = {}
    for i in range(0,len(factors)):
        j = 0
        if len(factors[i])>1:
            no_params = (2**len(factors[i]))/2
        else:
            no_params = 1
        pr = np.random.uniform(0,1,int(no_params))
        temp_dframe = pd.DataFrame(get_binary(factors[i].shape[0]),columns=factors[i])
        temp_dframe['pr'] = -1
        ptr = 0
        while j<temp_dframe.shape[0]:
            temp_dframe.loc[j,'pr'] = pr[ptr]
            temp_dframe.loc[j+1,'pr']= 1-pr[ptr]
            j+=2
            ptr+=1

        initial_prob[factors[i][-1]] = temp_dframe.copy()
    return initial_prob


def read_model_file(model_file_path):
    lines = []
    # read all the lines and remove empty lines from it
    for line in open(model_file_path,'r').read().split('\n'):
        if line.rstrip():
            lines.append(line)
    line_no = 0
    graph = nx.Graph()
    # capturing the network type
    network_type = lines[line_no]
    line_no+=1
    # capturing the number of variables
    no_of_variables = lines[line_no]
    line_no+=1
    # capturing the cardinalities of the variables
    cardinality_line = lines[line_no].rsplit()
    # storing cardinalities a as list of int
    variable_cardinalities = list(map(int,cardinality_line))
    line_no+=1
    # capturing number of cliques
    no_of_cliques = int(lines[line_no])
    line_no+=1
    factors = []
    cpt = []
    # dict_dframe = {}
    for i in range(no_of_cliques):
        cliques_input = lines[line_no+i]
        cliques = list(map(int,lines[line_no+i].rsplit()))[1:]
        # adding nodes to the graph
    #     print(cliques)
        graph.add_nodes_from(cliques)
        # check length of cliques if > 1 then add that edge
        if(len(cliques)>1):
            # if there are more than 2 nodes in the cliques then generate all combinations of pairs and add edge to the graph
            graph.add_edges_from(list(set(itertools.combinations(cliques, 2))))
        # append cliques to the factors list
        factors.append(np.array(cliques))

    line_no = line_no+i

    probab_dict = {}
    line_no_cp = line_no
    line_no = line_no+1
    ct = 0
    while(True):
        if line_no+1 >= len(lines):
            break
        var = int(lines[line_no])
        k=0
        cps = []

        binar = get_binary(int(np.log2(var)))
        cpt_data = pd.DataFrame(binar,columns=factors[ct])
        while k < var:
            filtered_line = " ".join(lines[line_no+1].split())
            fac_val = filtered_line.split(' ')
            cps.append(list(map(float,fac_val)))
            k = k + len(fac_val)
            line_no = line_no+1
        cpt_data['pr'] = np.array(cps).flatten()

        if (cpt_data['pr']==0).any():
            i=0
            while i < cpt_data.shape[0]:
                if (cpt_data.loc[i:i+1,'pr']==0).any():
                    if(cpt_data.loc[i,'pr']==0):
                        cpt_data.loc[i,'pr'] = 10**-5
                        cpt_data.loc[i+1,'pr']=(1-cpt_data.loc[i,'pr'])
                    else:
                        cpt_data.loc[i+1,'pr'] = 10**-5
                        cpt_data.loc[i,'pr']=(1-cpt_data.loc[i+1,'pr'])
                i+=2

        probab_dict[factors[ct][-1]] = cpt_data.copy()
        line_no = line_no+1
        ct+=1
    return probab_dict, factors, graph


def find_likelihood(test_data,prob_dict_values_cp,factors):
    mul_arr = np.ones(test_data.shape)
    for factor in factors:
        variable = factor[-1]
        assignments = test_data[:,factor]
        binary_scale = 2**np.arange(assignments.shape[1])[::-1]
        int_from_bin = assignments.dot(binary_scale)

        prob = np.take(prob_dict_values_cp[variable],int_from_bin)

        mul_arr[:,variable] = prob
    model_likelihood = np.sum(np.log10(mul_arr),axis=1)
    return model_likelihood


def find_likelihood(test_data,prob_dict_values_cp,factors):
    mul_arr = np.ones(test_data.shape)
    for factor in factors:
        variable = factor[-1]
        assignments = test_data[:,factor]
        binary_scale = 2**np.arange(assignments.shape[1])[::-1]
        int_from_bin = assignments.dot(binary_scale)

        prob = np.take(prob_dict_values_cp[variable],int_from_bin)

        mul_arr[:,variable] = prob
    model_likelihood = np.sum(np.log10(mul_arr),axis=1)
    return model_likelihood




def read_data(file):
    df = pd.read_csv(file,skiprows=1,header=None,delimiter=' ')
    df = df.dropna(axis=1)
    df = df.astype('str')
    return df






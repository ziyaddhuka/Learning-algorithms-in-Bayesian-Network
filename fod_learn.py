import pandas as pd
import numpy as np
from helper_functions import *


def fod_learn(df, prob_dict_values, factors,):
    temp_ = np.array(df.astype(int))
    new_prob = prob_dict_values.copy()
    weights = np.ones(temp_.shape[0])
    for factor in factors:
        variable = factor[-1]
        prob_arr = new_prob[variable]
        factor_assignments = temp_[:,factor].copy()
        if len(factor)<2:
            new_prob_arr = []
            idx = np.where(factor_assignments==0)[0]
            numerator = weights[idx].sum()
            denominator = weights.sum()
            pr = (numerator)/(denominator)
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


def fod_train(model_file_path,train_file_path,test_data_file):
    
    df = read_data(train_file_path)
    probab_dict,factors,_ = read_model_file(model_file_path)
    trained_dict = probab_dict.copy()
    
    initial_prob = initialize_probability(factors)
    prob_dict_values = {}
    for i in range(len(initial_prob)):
        prob_dict_values[initial_prob[i].columns[-2]] = initial_prob[i]['pr'].values

    
    true_prob_dict_values = {}
    for i in range(len(probab_dict)):
        true_prob_dict_values[probab_dict[i].columns[-2]] = probab_dict[i]['pr'].values
        
        
        
    trained_prob = fod_learn(df,prob_dict_values,factors)
    

    test_data = pd.read_csv(test_data_file,skiprows=1,header=None,delimiter=' ')
    test_data = test_data.dropna(axis=1)
    
    train_likelihood = find_likelihood(np.array(test_data),trained_prob,factors)
    test_likelihood = find_likelihood(np.array(test_data),true_prob_dict_values,factors)
    lldiff = np.sum(np.abs(test_likelihood-train_likelihood))
    print('----------------------------')
    print('log likelihood difference = ', lldiff)
    print('----------------------------')
    return lldiff
    
if __name__ == '__main__':
    model_file_path = 'hw5-data/dataset1/1.uai'
    train_file = 'hw5-data/dataset1/train-f-1.txt'
    test_data_file = 'hw5-data/dataset1/test.txt'
    lldiff = fod_train(model_file_path,train_file,test_data_file)
    
    print('----------------------------')
    print('log likelihood difference = ', lldiff)
    print('----------------------------')
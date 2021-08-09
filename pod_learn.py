import pandas as pd
import numpy as np
from helper_functions import *


def generate_new_data(df,question_mark_pos):
    full_data = []
    index = []
    index.append(0)
    temp_df = df.copy()
    i=0
    for data in temp_df:
        # temp_df = ndf.copy()
        if (data=='?').any():
            question_mark_arr = question_mark_pos[question_mark_pos[:,0]==i][:,1]
            combinations = get_binary(question_mark_arr.shape[0])
            k = data.copy()
            k = np.array(k)
            k = np.tile(k,(combinations.shape[0],1))
            k[:,question_mark_arr] = combinations
            full_data.append(k)
            index.append(index[-1]+combinations.shape[0])
        else:
            full_data.append(np.array(data))
            index.append(index[-1]+1)
        i=i+1
    full_data = np.concatenate(full_data)
    return full_data,index


def get_question_mark_pos(df):
    temp_df = df.stack()
    question_mark_pos = np.array(temp_df[(temp_df=='?')].index.tolist())
    return question_mark_pos


def new_e_step(temp_, prob_dict_values_cp, index, factors):
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

def new_maximisation(temp_ ,weights, factors,prob_dict_values):
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




def pod_learn(df, factors):
    question_mark_pos = get_question_mark_pos(df)
    full_data = np.array(df)
    full_data, index = generate_new_data(full_data,question_mark_pos)
    prob_dict_values = {}
    initial_prob = initialize_probability(factors)
    for i in range(len(initial_prob)):
        prob_dict_values[initial_prob[i].columns[-2]] = initial_prob[i]['pr'].values
    weights = 0
    data_copy = full_data.astype(int)
    new_prob_train = prob_dict_values.copy()
    for i in range(0,20):
        print("EM step = ",i)
        weights = new_e_step(data_copy,new_prob_train, index, factors)
        new_prob_train = new_maximisation(data_copy,weights,factors,new_prob_train)  
    return new_prob_train


def train_em(uai_file,training_file,test_data_file):
    probab_dict, factors, _ = read_model_file(uai_file)
    df = read_data(training_file)    
    true_prob_dict_values = {}
    for i in range(len(probab_dict)):
        true_prob_dict_values[probab_dict[i].columns[-2]] = probab_dict[i]['pr'].values

    lldiff_arr = []
    for i in range(0,5):
        new_prob_train = pod_learn(df,factors)

        test_data = pd.read_csv(test_data_file,skiprows=1,header=None,delimiter=' ')
        test_data = test_data.dropna(axis=1)
        train_likelihood = find_likelihood(np.array(test_data),new_prob_train,factors)
        test_likelihood = find_likelihood(np.array(test_data),true_prob_dict_values,factors)
        lldiff = np.sum(np.abs(test_likelihood-train_likelihood))
        print('----------------------------')
        print('log likelihood difference = ', lldiff)
        print('----------------------------')
        lldiff_arr.append(lldiff)


    lldiff_arr = np.array(lldiff_arr)
    ldiff_mean = np.mean(lldiff_arr)
    ldiff_std = np.std(lldiff_arr)
    prt_str =  "{:.7f} ± {:.7f}".format(ldiff_mean, ldiff_std)
    print('----------------------------')
    print("mean and standard deviation for log likelihood sum = ", prt_str)
    print('----------------------------')
    return





if __name__ == '__main__':
    model_file_path = 'hw5-data/dataset1/1.uai'
    train_file = 'hw5-data/dataset1/train-p-1.txt'
    test_data_file = 'hw5-data/dataset1/test.txt'
    train_em(model_file_path,train_file,test_data_file)
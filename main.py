#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 22:01:28 2021

@author: zad
"""

from fod_learn import fod_train
from pod_learn import train_em
from mixture_model import mixture_model
import sys


if __name__ == '__main__':
    uai_file = sys.argv[1]
    task = sys.argv[2]
    train_file = sys.argv[3]
    test_file = sys.argv[4]
    
    if int(task)==1:
        lldiff = fod_train(uai_file,train_file,test_file)
        
    elif int(task)==2:
        lldiff = train_em(uai_file,train_file,test_file)
    
    elif int(task)==3:
        try:
            k = int(sys.argv[5])
        except:
            print("please enter value of k")
        lldiff = mixture_model(uai_file,train_file,test_file,k)

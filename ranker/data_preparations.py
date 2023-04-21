import json

import numpy as np
import pandas as pd

hash_dict={}


def create_feature_vector(input_dir="generations",gen_num=200):
    global hash_dict
    data_set =[]
    for i in range(1,gen_num):
        df = pd.read_csv(f"../optimizer/{input_dir}/{i}.csv")
        source,target = df['source'],df['target']
        for i,j in zip(source,target):
            i = change_to_array(i)
            hashed_i = hashFloatArray(i)
            if hashed_i not in hash_dict.keys():
                hash_dict[hashed_i]= len(data_set)
                data_set.append(i)
            j = change_to_array(j)
            hashed_i = hashFloatArray(j)
            if hashed_i not in hash_dict.keys():
                hash_dict[hashed_i] = len(data_set)
                data_set.append(j)
    data=pd.DataFrame.from_records(data_set)
    data.to_csv(f"generations/{path}/features.csv",index=False)


def create_edge_vector(input_dir="generations",gen_num=200):
    global hash_dict
    data_set = []
    for i in range(1, gen_num):
        df = pd.read_csv(f"../optimizer/{input_dir}/{i}.csv")
        source, label,target = df['source'], df['label'],df['target']
        for num,j in enumerate(label):
            array_source = change_to_array(source[num])
            array_target = change_to_array(target[num])
            hashed_source=hashFloatArray(array_source)
            hashed_target = hashFloatArray(array_target)
            data_set.append([hash_dict[hashed_source],hash_dict[hashed_target],j])
    data = pd.DataFrame.from_records(data_set)
    data.to_csv("edges.csv")

def hashFloatArray(arr):
    h = ''
    for i in arr:
        n= hash(i)
        h+=str(n)
    return h

def change_to_array(i:str):
    i = i.split("[")[1].split("]")[0]
    i = np.fromstring(i, dtype=float, sep=' ')
    return i

def create_edge_vector_generation(generation_path):
    for i in range(1, 200):
        df = pd.read_csv(f"../optimizer/{generation_path}/{i}.csv")
        source, label, target = df['source'], df['label'], df['target']
        data_set = []
        for num, j in enumerate(label):
            array_source = change_to_array(source[num])
            array_target = change_to_array(target[num])
            hashed_source = hashFloatArray(array_source)
            hashed_target = hashFloatArray(array_target)
            data_set.append({"Src":hash_dict[hashed_source], "Dst":hash_dict[hashed_target], "Weight":j})
        data = pd.DataFrame.from_records(data_set)
        data.to_csv(f"generations/{path}/{i}.csv")

function_name ='Griewank'
dimension = 30
path = f"{function_name}/d{dimension}"
create_feature_vector(path)
create_edge_vector_generation(path)

# sample train/valid/test files for each client

import argparse
import os
import re
import random
import numpy as np


def parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_clients", type=int, help="num of clients", default=10)
    parser.add_argument("--input_dir", type=str, help="dir of file to be sampled")
    parser.add_argument("--output_dir", type=str) 
    args = parser.parse_args()

    return args

# filename = './wikitext-103-raw/wiki.test.raw'
# output_dir = './wikitext-103-raw/shuffle'

def sample(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_clients = args.number_clients

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    train_file_id, valid_file_id, test_file_id = get_file_id(args)  # get file id list
    
    # distribute files equally
    train_file_per_client = len(train_file_id)/num_clients
    valid_file_per_client = len(valid_file_id)/num_clients
    test_file_per_client = len(test_file_id)/num_clients

    # generate empty file list for each client 
    train_id, valid_id, test_id = [[]for i in range(num_clients)],[[]for i in range(num_clients)],[[]for i in range(num_clients)]
    #print(test_file_id)
    
    for client in range(num_clients-1):
        train_id[client].append(np.random.choice(train_file_id, size = int(train_file_per_client) , replace = False))
        valid_id[client].append(np.random.choice(valid_file_id, size = int(valid_file_per_client) , replace = False))
        test_id[client].append(np.random.choice(test_file_id, size = int(test_file_per_client) , replace = False))

        train_id[client] = train_id[client][0]
        valid_id[client] = valid_id[client][0]
        test_id[client] = test_id[client][0]

        # remove the assigned file id
        train_file_id = list(set(train_file_id).difference(set(train_id[client])))
        valid_file_id = list(set(valid_file_id).difference(set(valid_id[client])))
        test_file_id = list(set(test_file_id).difference(set(test_id[client])))
        
        #print(test_id[client])

    # the last client get the remaining files
    train_id[num_clients-1] = train_file_id
    valid_id[num_clients-1] = valid_file_id
    test_id[num_clients-1] = test_file_id
    #print(test_id[num_clients-1])

    write_line(args, train_id, valid_id, test_id)

        

def get_file_id(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_clients = args.number_clients
    train_dir = os.path.join(input_dir, "train_split")
    valid_dir = os.path.join(input_dir, "valid_split")
    test_dir = os.path.join(input_dir, "test_split")
    for dataset in  train_dir, valid_dir, test_dir:
        file_list = []
        for root, dirs, files in os.walk(dataset):
            for file in files:
                file_list.append(os.path.splitext(file)[0])
        if root.split('/')[-1] == "train_split":
            train_file_id = file_list
        elif root.split('/')[-1] == "valid_split":
            valid_file_id = file_list
        elif root.split('/')[-1] == "test_split":
            test_file_id = file_list

    return train_file_id, valid_file_id, test_file_id


def write_line(args, train_id, valid_id, test_id):
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_clients = args.number_clients
    
    train_dir = os.path.join(input_dir, "train_split")
    valid_dir = os.path.join(input_dir, "valid_split")
    test_dir = os.path.join(input_dir, "test_split")

    paragraph = []
    f_id, title_flag = 1, -1
    
    for client in range(num_clients):
        client_dir = os.path.join(output_dir, "client_"+str(client))
        if not os.path.exists(client_dir):
            os.mkdir(client_dir)

        train = os.path.join(client_dir, 'train.raw')
        valid = os.path.join(client_dir, 'valid.raw')
        test = os.path.join(client_dir, 'test.raw')
       
        train_list = train_id[client]
        valid_list = valid_id[client]
        test_list = test_id[client]

        for file in train_list:
            filename = os.path.join(train_dir, str(file) + ".raw")
            train_file = open(filename, "r", encoding="utf-8")
            write_file = open(train, 'a')
            for line in train_file:   
                write_file.writelines(line)
            train_file.close()
            write_file.close()

        for file in valid_list:
            filename = os.path.join(valid_dir, str(file) + ".raw")
            valid_file = open(filename, "r", encoding="utf-8")
            write_file = open(valid, 'a')
            for line in valid_file:
                write_file.writelines(line)
            valid_file.close()
            write_file.close()

        for file in test_list:
            filename = os.path.join(test_dir, str(file) + ".raw")
            test_file = open(filename, "r", encoding="utf-8")
            write_file = open(test, 'a')
            for line in test_file:
                write_file.writelines(line)
            test_file.close()
            write_file.close()
   

args = parameters()

def main():
    sample(args)
    #get_file_id(args)
    
if __name__ == "__main__":
    main()

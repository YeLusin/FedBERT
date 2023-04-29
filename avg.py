import os
import copy
import argparse
import torch
from fairseq.models.roberta import RobertaModel
from multiprocessing import Pool

def parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_client", type=int)  
    parser.add_argument("--input_dir", type=str)  
    parser.add_argument("--output_dir", type=str)  
    parser.add_argument("--current_epoch", type=int) 
    args = parser.parse_args()

    return args

# client_dir = "./fed/datasets_clients"
# output_dir = "./fed/datasets_clients"
# number_client = 2
# epoch = 1


def fed_avg(args):
    number_client = args.number_client
    client_dir = args.input_dir
    output_dir = args.output_dir
    epoch = args.current_epoch
    print(number_client, client_dir, output_dir, epoch)
    
    model_0 = torch.load(os.path.join(client_dir, "client_0/checkpoints/checkpoint_last.pt"), map_location=torch.device("cpu"))
    global_model_param = copy.deepcopy(model_0['model'])

    for client in range(1, number_client):
        checkpoint_dir = client_dir + "/client_"+str(client)+"/checkpoints"
        print("Loading: ", os.path.join(checkpoint_dir, "checkpoint_last.pt"))
        model = torch.load(os.path.join(checkpoint_dir, "checkpoint_last.pt"), map_location=torch.device("cpu"))
        for layer, param in model['model'].items():
            global_model_param[layer] += param

    for layer in global_model_param.keys():
        global_model_param[layer] /= number_client

    for client in range(0, number_client):
        checkpoint_dir = output_dir + "/client_"+str(client) +"/server"
        original_dir = client_dir + "/client_"+str(client)+"/checkpoints"
        model = torch.load(os.path.join(original_dir, "checkpoint_last.pt"), map_location=torch.device("cpu"))
        print(checkpoint_dir)

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            
        model['model'] = global_model_param
        torch.save(model, os.path.join(checkpoint_dir, "checkpoint_avg_"+str(epoch)+".pt"))
        print("Saving: ", os.path.join(checkpoint_dir, 'checkpoint_avg_'+str(epoch)+'.pt'))

args = parameters()
   
if __name__=='__main__':
    fed_avg(args)
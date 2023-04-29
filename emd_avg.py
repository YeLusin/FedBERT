# embedding average

import os
import copy
import argparse
import torch

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
    
    global_token_embedding = copy.deepcopy(model_0['model']["encoder.sentence_encoder.embed_tokens.weight"])
    global_position_embedding = copy.deepcopy(model_0['model']["encoder.sentence_encoder.embed_positions.weight"])
    
    # print("origal token_emd for model 1", model_0["model"]["encoder.sentence_encoder.embed_tokens.weight"])
    # print("origal position_emd for model 1", model_0["model"]["encoder.sentence_encoder.embed_positions.weight"])

    for client in range(1, number_client):
        checkpoint_dir = client_dir + "/client_"+str(client)+"/checkpoints"
        print("Loading: ", os.path.join(checkpoint_dir, "checkpoint_last.pt"))
        model = torch.load(os.path.join(checkpoint_dir, "checkpoint_last.pt"), map_location=torch.device("cpu"))
        global_token_embedding += model['model']["encoder.sentence_encoder.embed_tokens.weight"]
        global_position_embedding += model['model']["encoder.sentence_encoder.embed_positions.weight"]

        # print("origal token_emd for model 2", model["model"]["encoder.sentence_encoder.embed_tokens.weight"])
        # print("origal position_emd for model 2", model["model"]["encoder.sentence_encoder.embed_positions.weight"])

    global_token_embedding /= number_client
    global_position_embedding /= number_client

    # print(global_token_embedding)
    # print(global_position_embedding)

    for client in range(0, number_client):
        checkpoint_dir = output_dir + "/client_"+str(client) +"/server" 
        original_dir = client_dir + "/client_"+str(client)+"/checkpoints"
        model = torch.load(os.path.join(original_dir, "checkpoint_last.pt"), map_location=torch.device("cpu"))
        
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        model['model']["encoder.sentence_encoder.embed_tokens.weight"] = global_token_embedding
        model['model']["encoder.sentence_encoder.embed_positions.weight"] = global_position_embedding

        torch.save(model, os.path.join(checkpoint_dir, "checkpoint_avg_"+str(epoch)+".pt"))
        print("Saving: ", os.path.join(checkpoint_dir, 'checkpoint_avg_'+str(epoch)+'.pt'))

args = parameters()
   
if __name__=='__main__':
    fed_avg(args)
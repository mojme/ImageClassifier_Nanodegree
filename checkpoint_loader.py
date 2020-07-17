 #PROGRAMMER: Moritz Johannes Mey
# DATE CREATED: 16th of July 2020                             
# REVISED DATE: 
# PURPOSE: Loads a model from a checkpoint
import torch

def checkpoint_loader(filepath):
    """
    checkpoint = {"model": model,
                  "input_size" : input_layer,
                  "output_size": 102,
                  "hidden_layer_1": in_args.hidden_layers[0],
                  "hidden_layer_2": in_args.hidden_layers[1],
                  "drouput" : 0.2,
                  "learningrate" : in_args.learning_rate,
                  "class_to_idx": model.class_to_idx,
                  "last_layer_name": last_layer_name,
                  "model_fclassifier" : getattr(model, last_layer_name),
                  "optimizer" : optimizer, 
                  "state_dict" : model.state_dict()}
    """
    
    #load checkpoint and avoid cuda errors
    checkpoint = torch.load(filepath, map_location = ("cuda" if (torch.cuda.is_available()) else "cpu"))
    
    #load model
    model = checkpoint["model"]
    
    #asssign classifier
    last_layer_name = checkpoint["last_layer_name"]
        
    if last_layer_name == "fc": 
        model.fc = checkpoint["model_fclassifier"]
    elif last_layer_name == "classifier":
        model.classifier = checkpoint["model_fclassifier"]
    
    #state dict
    model.load_state_dict(checkpoint["state_dict"])
    #class to idx
    model.class_to_idx = checkpoint["class_to_idx"]
    
    return model
    


 #PROGRAMMER: Moritz Johannes Mey
# DATE CREATED: 15th of July 2020                             
# REVISED DATE: 
# PURPOSE: Predicts one of 102 flower types from an image

#imports
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim 
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
from PIL import Image
import json

#own imports (loader)
from predict_input_args import get_input_args  
from checkpoint_loader import checkpoint_loader
from process_image import process_image

def main():
    
    #Receive users commands
    in_arg = get_input_args()  
    
    """    
    in_arg gives:
    category_names='cat_to_name.json', checkpoint='checkpoint.pth', gpu=False, path_to_image='flowers/valid/13/image_05759.jpg',       top_k=3
    """
    
    #loading model 
    model = checkpoint_loader(in_arg.checkpoint)
    
    #building name dict
    with open(in_arg.category_names,"r") as f:
        cat_to_name = json.load(f)
        
    #convert image from path to numpy image
    np_im = process_image(in_arg.path_to_image)
    
    #using the GPU if activated in arg_in
    if in_arg.gpu == True:
        model.cuda()
        #convert np image to tensor
        torch_im = torch.from_numpy(np_im).type(torch.FloatTensor).cuda()
        
    else: 
        model.cpu()
        #convert np image to tensor
        torch_im = torch.from_numpy(np_im).type(torch.FloatTensor).cpu()
        
    #sizing
    torch_im.unsqueeze_(0)
    
    #geting logs with grads turned off
    with torch.no_grad():
        logps = model(torch_im)
    
    ps = torch.exp(logps)
    top_p, top_class_number = ps.topk (in_arg.top_k, dim=1)
    
    classes = list()
    
    #iterating through top_clas_number and appending the right name to classes
    for index in top_class_number[0]:
        #double for-loop to keep the right order in classes
        for key, m_idx in model.class_to_idx.items():
            if index == m_idx:
                classes.append(key)
                
    #now getting the right names from the class keys 
    
    for i in range(len(classes)):
        classes[i] = cat_to_name[classes[i]]
    
    #print(top_p)
    #print(classes)
    return top_p, classes
    

# Call to main function to run the program 
if __name__ == "__main__":
    main()

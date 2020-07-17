 #PROGRAMMER: Moritz Johannes Mey
# DATE CREATED: 15th of July 2020                             
# REVISED DATE: 
# PURPOSE: Retreive and parse command line arguments provided by the user when running the program in the terminal 
#  The user shall define himself:
#        1. The directory of the training & testing data AS data_directory
#        2. The directory to save date AS save_dir: save_directory
#        3. Choose an architecture from torchvision models AS arch: "model"
#        4. Set hyperparameters AS learning_rate, hidden units, epochs
#        5. Choose to use GPU for the training
    
#import modules
import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_directory", type = str, default = "flowers/", help = "path to image folder(s)")
    parser.add_argument("--save_dir", type = str, default = "save_directory/", help = "path to saving directory")
    parser.add_argument("--arch", type = str, default = "vgg16", help = "name of the CNN Model Architecture")
    parser.add_argument("--hidden_layers", type = int, default = [512, 256], help = "specifiy two hidden layers")
    #find help for nargs = n here: https://stackoverflow.com/questions/18924061/argparse-set-default-to-multiple-args
    #https://mkaz.blog/code/python-argparse-cookbook/
    parser.add_argument("--learning_rate", type = float, default = 0.003, help = "set learning rate")
    parser.add_argument("--epochs", type = int, default = 7, help = "set your training epochs")
    parser.add_argument("--gpu", type = bool, default = False , help = "enable gpu: True, disable gpu: False")
    
    return parser.parse_args()
    
    
 #PROGRAMMER: Moritz Johannes Mey
# DATE CREATED: 16th of July 2020                             
# REVISED DATE: 
# PURPOSE: Retreive and parse command line arguments provided by the user when running the program in the terminal 
#  The user shall define himself:
#        1.input a file to a single image
#        2.input a path to a checkpoint
#        3.input top_k number
#        4.choose the map of category names
#        5. turn on and off gpu
#import modules
import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path_to_image", type = str, default = "flowers/valid/13/image_05749.jpg", help = "path to image")
    parser.add_argument("--checkpoint", type = str, default = "save_directory/checkpoint.pth", help = "path to checkpoint")
    parser.add_argument("--top_k", type = int, default = 3, help = "give number of top classes")
    parser.add_argument("--category_names", type = str, default = "cat_to_name.json", help = "set your category name file")
    parser.add_argument("--gpu", type = bool, default = False , help = "enable gpu: True, disable gpu: False")
    
    return parser.parse_args()
    
    
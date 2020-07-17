 #PROGRAMMER: Moritz Johannes Mey
# DATE CREATED: 16th of July 2020                             
# REVISED DATE: 
# PURPOSE: processing and scaling a given image to hand it to a pytorch model

import numpy as np
from PIL import Image

def process_image(image):
    
    im = Image.open(image)
    im = im.resize((256,256))
    
    #crop the image to 224x224
    # The crop method from the Image module takes four coordinates as input.
    # The right can also be represented as (left+width)
    # and lower can be represented as (upper+height).
    #Source: PIL Library
    #left vertical drawn at 16, top horizontal also 16, right vertical at 240, bottom horizontal as well
    
    left = (256-224)/2
    top = left
    right = (256-224)/2 +224
    bottom = right
    
    im = im.crop((left, top, right, bottom))
    
    np_image = np.array(im)/255
    
    means = np.array([0.485, 0.456, 0.406])
    st_dev =  np.array([0.229, 0.224, 0.225])
    
    #normalising in the required way
    np_image = (np_image-means) / st_dev
    
    #color channel needs to be first:
    np_image = np_image.transpose(2,0,1)
    
    return np_image
    
    
    
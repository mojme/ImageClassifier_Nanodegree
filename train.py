 #PROGRAMMER: Moritz Johannes Mey
# DATE CREATED: 15th of July 2020                             
# REVISED DATE: 
# PURPOSE: Trains an image classifier to classify flowers


#Import python modules here

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim 
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
import pathlib

#import own modules
from get_input_args import get_input_args 

#main program function defined below
def main():
    """"Requirements: 
    This program shall train the neural network.
    The user shall define himself:
        1. The directory of the training & testing data AS data_directory
        2. The directory to save date AS save_dir: save_directory
        3. Choose an architecture from torchvision models AS arch: "model"
        4. Set hyperparameters AS learning_rate, hidden units, epochs
        5. Choose to use GPU for the training
    
    The program will then print out: Training Loss, Validation Loss & Validation Accuracy
    
    """
    
    #Receive users commands
    in_arg = get_input_args()
    
    """
    in_arg default gives: 
    (arch = "vvg16", data_directory="flowers/", epochs=7, gpu=False, hidden_layers = [512,256], learning_rate = 0.003
    save_dir = "save_directory/)
    """
    
    #Now: 
    #Check whether files are in the user directory and load the files if available
    data_dir = in_arg.data_directory
    
    has_train = any(pathlib.Path(data_dir + "train/").iterdir())
    has_test = any(pathlib.Path(data_dir + "test/").iterdir())
    has_valid = any(pathlib.Path(data_dir + "valid/").iterdir())
    
    if has_train and has_test and has_valid:
        train_dir = data_dir + 'train/'
        valid_dir = data_dir + 'valid/'
        test_dir = data_dir + 'test/'
    else:
        print("The given directory, " ,data_dir, " does have an empty train/ valid/ or test/ folder. Now using flowers/")
        data_dir = "flowers/"
    
    #Defining transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255), 
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255), 
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # Loading the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform = validation_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_dataset,batch_size = 64)
    validationloader = torch.utils.data.DataLoader(validation_dataset,batch_size = 64)
    
    """Loading Data Completed"""
    
    #Now checking the given model
    #   a) check - is among models?
    #   b) check input required for the model
    #   c) check whether its model.fc or model.classifier
    
    #a)
    if in_arg.arch in models.__dict__:
        model = getattr(models, in_arg.arch)(pretrained = True)
    else:
        print("Model not in torchvision.models, now using VGG16")
        #but: the show must go on
        model = models.vgg16(pretrained=True)
        
    #b )
    last_layer_name = ""
    if "fc" in model.__dict__["_modules"]:
        last_layer_name = "fc"
    elif "classifier" in model.__dict__["_modules"]:
        last_layer_name = "classifier"
    
    #c test the type and accordingly find out how many input layers the model has
    input_layer = 0
    if isinstance(getattr(model, last_layer_name), torch.nn.modules.container.Sequential):
        #if models last layer is sequential 
        input_layer = getattr(model,last_layer_name)[0].in_features
    elif isinstance(getattr(model, last_layer_name), torch.nn.modules.linear.Linear):
        #if models last layer is linear
        input_layer = getattr(model,last_layer_name).in_features
    
    #building the given model with given specs
    
    #freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    #set up fc or classifier --> fclassifier
    fclassifier = nn.Sequential(nn.Linear(input_layer,in_arg.hidden_layers[0]),
                                nn.ReLU(),
                                nn.Dropout(p = 0.2),
                                nn.Linear(in_arg.hidden_layers[0],in_arg.hidden_layers[1]),
                                nn.ReLU(),
                                nn.Dropout(p = 0.2),
                                nn.Linear(in_arg.hidden_layers[1],102),
                                nn.LogSoftmax(dim=1))
    #assign fclassifier
    if last_layer_name == "fc": 
        model.fc = fclassifier
    elif last_layer_name == "classifier":
        model.classifier = fclassifier
    
    #checking if cuda available and if gpu shall be used
    if torch.cuda.is_available and in_arg.gpu == True:
        model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    #to be informed: print("using ", device)
    
    #set criterion and optimm with learnrate as given
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(getattr(model,last_layer_name).parameters(), lr = in_arg.learning_rate)
    model.to(device)
    
    #defining the training process
    
    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    #when do you want the training loss to be printed?
    print_every = 20
    #when shall the model start testing regularly?
    test_every = 100
    
    model.train()
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps +=1 
            
            #move to GPU if possible
            
            images, labels = images.to(device), labels.to(device)
            
            #train:
            
            #zero grad:
            optimizer.zero_grad()
            
            #get probabilities:
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            #train printer
            if steps % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}.."
                      f"Step {steps}.."
                      f"Train loss: {running_loss/print_every:.3f}..")
                running_loss = 0
            
            #testing printer
            if  epoch > 2 and steps % test_every == 0:
                print("\n ...Testing Initiated ... \n")
                print("Step", steps)
            
                test_loss = 0
                accuracy = 0
                model.eval()
            
                with torch.no_grad():
                    for inputs, labels in testloader: 
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs) #or model.forward(inputs)
                        batch_loss = criterion(logps,labels)
        
                        test_loss += batch_loss.item()
        
                        #accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk (1, dim =1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
                print(f"Test loss: {test_loss/len(testloader):.3f}.."
                      f"Test accuracy: {accuracy/len(testloader):.3f}"
                      "\n ...Testing Stopped... \n")
                #set model back to training mode
                model.train()
            

    print("I'm done with the training")
    
    #training and testing finished
    
    # testing again
    print("\n ...Final testing Initiated ... \n")
    print("Total steps", steps)
          
    test_loss = 0
    accuracy = 0
    model.eval()
           
    with torch.no_grad():
        for inputs, labels in testloader: 
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs) #or model.forward(inputs)
            batch_loss = criterion(logps,labels)
            test_loss += batch_loss.item()
        
            #accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk (1, dim =1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(f"Test loss: {test_loss/len(testloader):.3f}.."
              f"Test accuracy: {accuracy/len(testloader):.3f}"
              "\n ...Testing Stopped... \n")
    model.train()
            
    #now: saving checkpoint
    
    model.class_to_idx = train_dataset.class_to_idx
    
    checkpoint = {"model": model,
                  "input_size" : input_layer,
                  "output_size": 102,
                  "hidden_layer_1": in_arg.hidden_layers[0],
                  "hidden_layer_2": in_arg.hidden_layers[1],
                  "drouput" : 0.2,
                  "learningrate" : in_arg.learning_rate,
                  "class_to_idx": model.class_to_idx,
                  "last_layer_name": last_layer_name,
                  "model_fclassifier" : getattr(model, last_layer_name),
                  "optimizer" : optimizer, 
                  "state_dict" : model.state_dict()}
    torch.save(checkpoint, in_arg.save_dir + "checkpoint.pth")
    print("Model saved sucessfully under: ", in_arg.save_dir, "checkpoint.pth")
                  
   
# Call to main function to run the program 
if __name__ == "__main__":
    main()
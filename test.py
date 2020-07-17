 #PROGRAMMER: Moritz Johannes Mey
# DATE CREATED: 16th of July 2020                             
# REVISED DATE: 
# PURPOSE: a file to test the module and to see how it predicts

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim 
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from checkpoint_loader import checkpoint_loader

def main():
    data_dir = "flowers/"
    
    train_dir = data_dir + 'train/'
    valid_dir = data_dir + 'valid/'
    test_dir = data_dir + 'test/'
    
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

    #    Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_dataset,batch_size = 64)
    validationloader = torch.utils.data.DataLoader(validation_dataset,batch_size = 64)
    
    """Loading Data Completed"""
    
    """ Now loading checkpoint"""
    # TODO: Write a function that loads a checkpoint and rebuilds the model

    model = checkpoint_loader("save_directory/checkpoint.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.002)
    
    print("\n ...Testing Initiated ... \n")
    test_loss = 0
    accuracy = 0
    steps = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader: 
            steps = steps+1
            print(steps, "/", len(testloader))
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs) #or model.forward(inputs)
            batch_loss = criterion(logps,labels)
            test_loss += batch_loss.item()
        
            #accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk (1, dim =1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            print(f"current accuracy: {accuracy/len(testloader):.3f}")
        
        print("...done testing...")
        print(f"Test loss: {test_loss/len(testloader):.3f}.."
              f"Test accuracy: {accuracy/len(testloader):.3f}"
              "\n ...Testing Stopped... \n")
    model.train()
            

    print("I'm done with the training")
    
 
   
# Call to main function to run the program 
if __name__ == "__main__":
    main()
# ImageClassifier_Nanodegree
Image Classifier built within the Udacity Nanodegree A.I. Programming

Train.py
This program shall train the neural network.
    The user shall define himself:
        1. The directory of the training & testing data AS data_directory
        2. The directory to save date AS save_dir: save_directory
        3. Choose an architecture from torchvision models AS arch: "model"
        4. Set hyperparameters AS learning_rate, hidden units, epochs
        5. Choose to use GPU for the training
        Every value has a default value specified in get_input_args.py
   The program will then print out: Training Loss, Validation Loss & Validation Accuracy
   And it will save the model under the the given save directory (or default) under the name "checkpoint.pth"
    

get_input_args.py
Parses through the arguments
The user shall define himself:
        1. The directory of the training & testing data AS data_directory
        2. The directory to save date AS save_dir: save_directory
        3. Choose an architecture from torchvision models AS arch: "model"
        4. Set hyperparameters AS learning_rate, hidden units, epochs
        5. Choose to use GPU for the training
 Also has a default for everything. 
 the hidden model should take a max. of two layers, it might lead to issues in train.py whilst saving the checkpoint otherwise
 (to have more hidden layers, saving the layers could be done in a list and loading them with a loop going through the list)

test.py
Is an independent file to load a model from a checkpoint defined in the code and test it against some images defined in the code. 

predict.py
predict loads the checkpoint.pth from a given directory. --> the directory here is only the folder. The program expects the file to be called checkpoint.pth
the loading of the checkpoint is done with checkpoint_loader.py
the user can define inputs, user inputs are defined in predict_input_args
predict.py uses a json file called, specified in predict_input_args to read out the category names and convert category numbers/indexes to names for the output

checkpoint_loader.py
loads a model from the checkpoint, can only work with two hidden layers, defines the output size fix as 102 and a fix dropout of 0.2

predict_input_args.py
Retreive and parse command line arguments provided by the user when running the program in the terminal 
 The user shall define himself:
       1.input a file to a single image
       2.input a path to a checkpoint
       3.input top_k number
       4.choose the map of category names
       5. turn on and off gpu


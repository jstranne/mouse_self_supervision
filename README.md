# Mouse Self Supervision

Note: If looking for comments, the files associated with the RP (relative positioning) task are commented the most. 
## Important Files
* list_training_files.py: This file creates a txt file list of all of the mice that are part of the training data.
* Extract_Data.py: A file full of helper methods. Some of these methods include ones to read in the data, methods to filter the data, and methods to window the data.
* Create_Windowed_Dataset.py: This file should be run to create the windowed dataset. It takes all of the names that are listed in the `training_names.txt` folder and creates a new windowed dataset for each mouse.
* Stager_net_pratice.py: Holds the basic structure for the embedder (called a StagerNet). Adapted from https://arxiv.org/pdf/2007.16104.pdf
* Relative_Positioning.py, Temporal_Shuffling.py, CPC_Net.py: These files hold the network structure for the 3 pretext tasks embedders.
* RP_data_loader.py, TS_data_loader.py, CPC data_loader.py: All of theseare responsible for running the code to train the 3 embedders, using the different pretext tasks. These also hold the important code for how the embedders are trained, as well as how positive and negative samples for the pretext tasks are created.
* RP_Downstream_Trainer.py: This is the main file that holds all of the methods for the downstream training. Ideally you would never have to run this and should instead call its methods from the Jupyter notebooks which are easier to understand and use. 
* ... down_onevsall notebooks: These are used to run the final downstream training loops. They are easy to use notebooks that should work out of the box. Please see the note in the `My Workflow` section on why there are so many down_onevsall notebooks with different prefixes.
* models folder: This folder contains all of the saved stagernet model weights that can be loaded for quick use.

## My workflow
* First run the list_training_files.py file. This will create a list of all of the mice that have data stored in the Mouse_Training_Data folder. In this folder, there is an INT_TIME folder and a LFP_Data folder (containig the times different classes occur at and the local field potential data). This run will create a `training_names.txt` file containing the prefix name for every mouse.
* Run Create_Windowed_Dataset.py: This will take care of breaking the data up into windows and assigning the proper labels to the data as well. It will create 4 files for each mouse, one holding an array of the windowed data, one having the labels, one with the start time of each window, and a final array with the windowed data not excluding the times the mice were being moved from place to place (for the pretext task).
* RP_data_loader.py, TS_data_loader.py, CPC data_loader.py: Run these to train the embedders based on the pretext tasks. Make sure to set the parameters for the model (such as tpos and tneg). The learned embedder will be saved with a file such as `RP_stagernet_{tpos_val}_{tneg_val}.pth`. Obviously, this will change if the TS or CPC task is used.
* Now we have the learend embedder that we can use on the downstream task.
* RP_Trainer_Down_Onevsall: The Onevsall trainer files should be used to do the actual training. They will actually used a grouped k fold approach to the train test split. There will be 5 splits. In addition, the number of possitive classes that are used for training will be a list like 1, 10, 100, 1000, ..., None. The last non None member of this list will be determined by the smallest_class_len method. None means that all the data will be used for downstream training. Since there are 5 folds, all the trials will be run 5 times and then the average accuracy on the test set will be reported as a dictionary (keys are the number of positive samples per class, value is the average test accuracy).
* Other down_onevsall notebooks: There are many other notebooks with similar suffixes that perform the same way as the one above. They obviously should be combined into a single notebook that takes the type of model to train with as a parameter, but are currently separated so I could keep track of the results and run them at the same time. Random trainer and fully supervised are a little different. Random uses no trained embedder and fixes the weights of a random embedder. The fully supervised trainer starts with an untrained embedder, but allows its weights to be trainable so the entire network is trained with the supervised data only. Make sure to change the path of the stagernet model if you want to change to use a different pretrained embedder.
* All3_Trainer_Downstream and Just2_Trainer_Downstream: These use 2 or 3 stagernets as the embedders before the final trained layer. Otherwise, they work the same as above.



import os


print("Creating List of Training Files")
directory=os.listdir('.'+os.sep+'Mouse_Training_Data'+os.sep+'LFP_Data')
f = open('training_names.txt','w')
for folder in directory:
    print(folder)
    if folder.startswith('MouseCK'):
        f.write(folder[:folder.index("_LFP.mat")]+'\n')
f.close()
print("Done")

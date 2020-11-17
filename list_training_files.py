import os


print("Creating List of Training Files")
directory=os.listdir('.'+os.sep+'Mouse_Training_Data'+os.sep+'LFP_Data')
f = open('training_names.txt','w')
mouse_list=[]
for folder in directory:
    print(folder)
    if folder.startswith('MouseCK'):
        mouse_list.append(folder[:folder.index("_LFP.mat")])
        #f.write(folder[:folder.index("_LFP.mat")]+'\n')
#f.close()
mouse_list.sort()




SKIP_MICE = True # shoud i skip the second trial of the same mice

last = '_'
for file_name in mouse_list:
    if SKIP_MICE:
        if last[:last.index("_")] != file_name[:file_name.index("_")]:
            f.write(file_name+'\n')
        last = file_name
    else:
        f.write(file_name+'\n')
f.close()

print("Done")

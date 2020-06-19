import os
path_to_images = './SunHat/Sun hat/Images/'   
file_path = './SunHat/Sun hat/Images/train.txt'
f = open(file_path,'w+')
for each in os.listdir(path_to_images):
  if each.endswith('.jpg'):
    # print(os.path.join(path_to_images,each))
    f.write(os.path.join(path_to_images,each)+'\n')
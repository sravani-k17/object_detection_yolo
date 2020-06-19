import shutil
import os

source = './Ciphense/DataCollection/'
dest1 = './Ciphense/Annotations/'
new_folder='./Ciphense/finalData/'
files_1 = os.listdir(dest1)
labels = []
for i in files_1:
   name,ext = os.path.splitext(i)
   labels.append(name)

files = os.listdir(source)
for f in files:
   name, ext = os.path.splitext(f)
   if name in labels:
       shutil.copy(source+f, new_folder)
import shutil
import os

#source = './Ciphense/DataCollection/'
source1 = './main_valid/'
new_folder='./main_train/yolo/'

#new_folder='./Ciphense/finalData/'
files_1 = os.listdir(source1)



labels=[]
for i in files_1:
	name,ext = os.path.splitext(i)
	if(ext==".jpg"):
		labels.append(name)
		shutil.copy(source1+i,new_folder)


files = os.listdir(source1)
count=0
for f in files:
	name, ext = os.path.splitext(f)
	if(ext==".xml"):
		if name in labels:
			count+=1
			shutil.copy(source1+f,new_folder)
			



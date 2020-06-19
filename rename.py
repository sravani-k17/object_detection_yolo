import os
from PIL import Image
path = './Ciphense/DataCollection/'
count=0
for each in os.listdir(path):
   # if each.endswith('jpg'):
   # tag = each.split('.')[-1]
   count+=1
   img = Image.open(path+each).convert("RGB")
   img.save(path+'hairband_{:04}'.format(count)+'.jpg','jpeg')

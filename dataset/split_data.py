import os
from shutil import copy
import random

# 若file不存在，则创建file
def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


file = 'oubaodata'
flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]
mkfile('oubaodata/train')
for cla in flower_class:
    mkfile('oubaodata/train/'+cla)

mkfile('oubaodata/val')
for cla in flower_class:
    mkfile('oubaodata_data/val/'+cla)

split_rate = 0.1
for cla in flower_class:
    cla_path = file + '/' + cla + '/'
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'oubaodata/val/' + cla
            copy(image_path, new_path)
        else:
            image_path = cla_path + image
            new_path = 'oubaodata/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    print()

print("processing done!")

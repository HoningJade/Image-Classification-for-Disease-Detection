# -*- coding: utf-8 -*-
# Author : kaswary
# Time   : 2020/3/22 12:22


#提取目录下所有图片,更改尺寸后保存到另一目录
from PIL import Image
import os.path
import glob

def convertjpg(jpgfile, outdir, width=256, height=256):
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

if __name__ == '__main__':
    for jpgfile in glob.glob(r"D:/machine_learning/vgg/data/validation_data/1/*.jpg"):
        convertjpg(jpgfile, "D:/machine_learning/vgg/data/valid/1/")

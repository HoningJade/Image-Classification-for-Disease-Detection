import os
import re

data_root = "D:/deep-learning-for-image-processing-master/data/train_oubao/ill/"
folder = os.listdir(data_root)
# folder = ['6', '16', '19', '20', '23', '25', '39', '47', '54', '56', '59', '79', '81', '77', '85', '87', '91', '93', '94', '101', '104', '105', '106', '110', '120', '123']

k = 0
for m in folder:
    path = data_root + folder[k] + '/'
    # 获取该目录下所有文件，存入列表中
    fileList = os.listdir(path)
    n = 0
    for i in fileList:
        # 设置旧文件名（就是路径+文件名）
        oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符

        # 设置新文件名
        newname = path + os.sep + 'a' + str(n + 100) + folder[k] + '.jpg'
        # newname = path + os.sep + 'a' + '.jpg'
        os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
        print(oldname, '======>', newname)
        n += 1
    k += 1



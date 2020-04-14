#统计一下数据中每个类别的数量


import sys

import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import os


SIXray_CLASSES = ('gun', 'knife', 'wrench', 'pliers', 'scissors')



class_to_ind = dict(zip(SIXray_CLASSES, range(len(SIXray_CLASSES))))


count = np.zeros(5)


root = '/media/dsg3/datasets/SIXray/Annotation'

listdir = os.listdir(root)

for name in listdir:
    print (name)
    tree = ET.parse(os.path.join(root, name))
    for obj in tree.findall('object'):
        xx = obj.find('name').text.lower().strip()    
        count [class_to_ind[xx]] += 1


print (count)
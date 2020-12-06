import os
import random


listdir = os.listdir('/media/dsg3/datasets/SIXray/Annotation')

test = random.sample(listdir, 200)

train = [x for x in listdir if x not in test]


with open('dataset-train.txt', 'w') as f:
    for item in train:
        f.writelines('{0}\n'.format(os.path.splitext(item)[0]))

with open('dataset-test.txt', 'w') as f:
    for item in test:
        f.writelines('{0}\n'.format(os.path.splitext(item)[0]))
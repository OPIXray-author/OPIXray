import os
import xml.etree.ElementTree as ET
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

'''
SIXray_CLASSES = (
    '带电芯充电宝', '不带电芯充电宝', '劣质充电宝', '其它充电宝',
    '金属打火机' '非金属打火机', '点烟器', '异形打火机', '镁棒', '其它火种',
    '扳手', '螺丝刀', '钳子', '其他日常工具',
    '管制刀具', '直刀', '折叠刀', '异形刀', '剪刀', '其它刀具',
    '电击器', '手铐', '警鞭', '弩', '其它军警器具',
    '手枪', '长枪', '仿真枪', '枪支部件', '其它枪支',
    '子弹', '空包弹', '信号弹', '其它弹药',
    '液化石油气', 'ZIPPO油', '打火机气', '压缩气体瓶', '其它危险品',
    '烟花', '礼花', '鞭炮', '烟饼（包）', '其它烟火制品',
    '雷管', '拉火管', '导火索', '炸药', '爆炸装置', '其它爆炸装置'
)
'''
# note: if you used our download scripts, this should be right
#SIXray_ROOT = '/media/dsg3/datasets/SIXray'
#SIXray_ROOT = '/media/dsg3/datasets/Xray20190704/'
SIXray_ROOT = '/media/dsg3/datasets/Xray20190723/'

type_dict = {}
type_sum_dict = {}

class_path = SIXray_ROOT + "class.xml"
tree = ET.parse(class_path)
class_root = tree.getroot()
for child in class_root:
    one_dict = {}
    type_name = None
    find_name = 0
    for grandson in child:
        if find_name == 0:
            find_name = 1
            type_name = grandson.text
        else:
            one_dict[grandson.text] = 0
    type_dict[type_name] = one_dict
#print(type_dict)

all_file = 0
for i in range(1):
    for j in range(1):
        dataset_train = open(SIXray_ROOT + "classify_train_test/battery_2cv_new_train.txt", "w", encoding='utf-8')
        dataset_test = open(SIXray_ROOT + "classify_train_test/battery_2cv_new_test.txt", "w", encoding='utf-8')
        train_or_set = 0
        test1000 = 0
        #遍历Annotation
        root_annotation = '/media/dsg3/datasets/Xray20190723/Anno_battery_2_version/'
        res = []
        #for root, dirs, files in os.walk(root_annotation):
            #for file in files:
        for i in range(1,50000, 1):
            for j in range(1):
                file_name = 'battery' + str(0)*(8-len(str(i))) +str(i) + '.txt'
                #txt_path = os.path.join(root, file)
                txt_path = root_annotation + file_name
                image_path = txt_path.replace('Anno_battery_2_version', 'Image_battery_2_version')
                image_path = image_path.replace('.txt', '.tiff')
                image_path1 = image_path.replace('.tiff', '.TIFF')
                if cv2.imread(image_path) is None:
                    if cv2.imread(image_path1) is None:
                        continue
                #print(cv2.imread(image_path))
                #print(cv2.imread(image_path1))

                #print(txt_path)
                # print(txt_path)
                if len(txt_path) > 4 and txt_path[-4:] == '.txt':
                    #不读H和V
                    if txt_path[-5] == 'H'or txt_path[-5] == 'V':
                        continue
                    # print("is .txt")
                    img = cv2.imread(image_path)
                    img_cut = image_path
                    if img is None:
                        img = cv2.imread(image_path1)
                        img_cut = image_path1
                    if img is None:
                        continue
                    #没有
                    height, width, channels = img.shape
                    #print("width")
                    with open(txt_path, "r", encoding='utf-8') as f1:
                        dataread = f1.readlines()
                        if len(dataread) == 0 or dataread[0] == '':
                            continue
                        haveuse = 0
                        tttemp = 0
                        #下面写的有问题!!!!!!!!!!!!!!重新分一下类吧！
                        for annotation in dataread:
                            bndbox = []
                            temp = annotation.split()
                            name = temp[1]
                            xmax = temp[4]
                            #print(xmax)
                            for key1 in type_dict.keys():
                                for key2 in type_dict[key1].keys():
                                    if key2 == name:
                                        type_dict[key1][key2] += 1
                            # 只读两类
                            if name == '带电芯充电宝' or name == '不带电芯充电宝':
                                if int(xmax) < width/2:
                                    haveuse = 1
                                    tttemp += 1
                        if haveuse == 0:
                            continue
                        all_file += 1
                        train_or_set = ((train_or_set + 1) % 4)
                        if train_or_set != 0 or test1000 >= 2000:
                            dataset_train.writelines(file_name[:-4] + '\n')
                        else:
                            test1000 += 1
                            #test1000 += tttemp
                            dataset_test.writelines(file_name[:-4] + '\n')
                        #cut_ = img[0 : int(height/2), 0 : int(width/2)]
                        #cut_ = img[0: int(height), 0: int(width)]
                        cut_ = img[0: int(height), 0: int(width / 2)]
                        cut_save = img_cut.replace('Image_battery_2_version', 'Image_battery_2cv')
                        cut_save = cut_save.replace('tiff', 'jpg')
                        cut_save = cut_save.replace('TIFF', 'jpg')
                        print(cut_save)
                        cv2.imwrite(cut_save, cut_)

                    #print(txt_path)

        dataset_train.close()
        dataset_test.close()
        for key1 in type_dict.keys():
            _sum = 0
            for value2 in type_dict[key1].values():
                _sum += value2
            type_sum_dict[key1] = _sum
        #print(res)
        print(type_dict)
        print(type_sum_dict)
classification = []
frequency = []
for key1 in type_sum_dict.keys():
    classification.append(key1)
    frequency.append(type_sum_dict[key1])
for i in range(len(frequency)-1):
    maxf = i
    for j in range(i+1, len(frequency)):
        if frequency[maxf] < frequency[j]:
            maxf = j
    if maxf != i:
        temp = classification[i]
        classification[i] = classification[maxf]
        classification[maxf] = temp
        temp = frequency[i]
        frequency[i] = frequency[maxf]
        frequency[maxf] = temp
for i in range(len(frequency)):
    print(classification[i] + ": " + str(frequency[i]))
print("半V视角一共找出图片张树：" + str(all_file))
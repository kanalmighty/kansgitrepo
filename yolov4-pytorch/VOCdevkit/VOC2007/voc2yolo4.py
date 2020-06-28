import os
import random

train_xmlfilepath='D:\\datasets\\voc\\VOCtrainval_06-Nov-2007\\VOCdevkit\\VOC2007\\Annotations'
saveBasePath='D:\\PyCharmSpace\\kansgitrepo\\yolov4-pytorch\\VOCdevkit\\VOC2007\\ImageSets\\Main'
all_xmlfilepath='D:\\datasets\\voc\\all_annotations\\Annotations'
trainval_percent=1
train_percent=0.9

train_xml_path = os.listdir(train_xmlfilepath)
all_xml_path = os.listdir(all_xmlfilepath)
total_xml = []
train_xml = []
for xml in train_xml_path:
    if xml.endswith(".xml"):
        train_xml.append(xml)

for xml in all_xml_path:
    if xml.endswith(".xml"):
        total_xml.append(xml)

all_num=len(total_xml)
train_num=len(train_xml)
list=range(all_num)
tv=int(train_num*trainval_percent)
tr=int(tv*train_percent)
all_id = [i.split('.')[0] for i in total_xml]
trainval_id = [i.split('.')[0] for i in train_xml]
trainval= random.sample(trainval_id,tv)
train=random.sample(trainval,tr)

print("train and val size",tv)
print("traub suze",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')

for i  in all_id:
    if i in trainval:
        ftrainval.write(i+'\n')
        if i in train:
            ftrain.write(i+'\n')
        else:
            fval.write(i+'\n')
    else:
        ftest.write(i+'\n')

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()

from xml.etree import cElementTree as ET
from PIL import Image
import os


"""
ImageID: [[ClassName, Confident, Xmin, Xmax, Ymin, Ymax]]
"""

if __name__ == '__main__':
    classlist = {}
    c = open('/home/palm/class-descriptions-boxable.csv').readlines()
    for elm in c:
        elm = elm[:-1].split(',')
        classlist[elm[0]] = elm[1]
    del c

    objlist = {}
    instantlist = open('/home/palm/train-annotations-bbox.csv').readlines()
    for elm in instantlist[1:]:
        elm = elm[:-1].split(',')
        if classlist[elm[2]].lower() in ['helmet', 'vehicle registration plate', 'boot', 'person', 'footware']:
            if elm[0] not in objlist:
                objlist[elm[0]] = []

            objlist[elm[0]].append([classlist[elm[2]], *elm[3:8], int(elm[8])+int(elm[10])+int(elm[11])])
    del instantlist

    rootpath = '/media/palm/data/ppa/v5/anns/train/'
    impath = '/media/palm/data/ppa/v5/images/train/'
    for key in objlist:
        anns = objlist[key]
        if os.path.exists(os.path.join(rootpath, key+'.xml')):
            continue
        if not os.path.exists(os.path.join(impath, key+'.jpg')):
            continue
        try:
            image = Image.open(impath+key+'.jpg')
            width, height = image.size
            root = ET.Element('annotation')
            ET.SubElement(root, 'filename').text = key+'.jpg'
            ET.SubElement(root, 'path').text = rootpath+key+'.jpg'
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(width)
            ET.SubElement(size, 'height').text = str(height)
            for ann in anns:
                if ann[-1] > 0:
                    continue
                obj = ET.SubElement(root, 'object')
                if ann[0].lower() in ['boot', 'footware']:
                    ann[0] = 'goodshoes'
                if ann[0].lower() == 'vehicle registration plate':
                    ann[0] = 'LP'
                ET.SubElement(obj, 'name').text = ann[0]
                bndbx = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbx, 'xmin').text = str(int(float(ann[2])*width))
                ET.SubElement(bndbx, 'xmax').text = str(int(float(ann[3])*width))
                ET.SubElement(bndbx, 'ymin').text = str(int(float(ann[4])*height))
                ET.SubElement(bndbx, 'ymax').text = str(int(float(ann[5])*height))
            tree = ET.ElementTree(root)
            tree.write(rootpath+key+'.xml')
        except FileNotFoundError:
            print(key)

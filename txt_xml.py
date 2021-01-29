# -*- coding: utf-8 -*-
import os,shutil
import cv2
from lxml.etree import Element, SubElement, tostring
def txt_xml(img_path,txt_path,xml_path):
    #读取txt的信息
    clas=[]
    img=cv2.imread(img_path)
    imh, imw = img.shape[0:2]
    txt_img=txt_path
    with open(txt_img,"r") as f:
        next(f)
        for line in f.readlines():
            line = line.strip('\n')
            list = line.split(" ")
            # print(list)
            clas.append(list)
    count = 0
    for h in range(1059):
        for k in range(len(clas)):
            print('-----')
            print(h)
            if int(clas[k][0])==h:
                if count==h:
                    if int(clas[k][0])<10:
                        img_xml = '00000' + clas[k][0] + '.xml'
                    elif ((int(clas[k][0])>=10) and (int(clas[k][0])<100)):
                        img_xml = '0000' + clas[k][0] + '.xml'
                    else:
                        img_xml = '000' + clas[k][0] + '.xml'
                    node_root = Element('annotation')
                    node_folder = SubElement(node_root, 'folder')
                    node_folder.text = '0019' #修改
                    node_filename = SubElement(node_root, 'filename')
                    # 图像名称
                    if int(clas[k][0])<10:
                        node_filename.text = '00000'+clas[k][0]
                    elif ((int(clas[k][0])>=10) and (int(clas[k][0])<100)):
                        node_filename.text = '0000'+clas[k][0]
                    else:
                        node_filename.text = '000'+clas[k][0]
                    node_size = SubElement(node_root, 'size')
                    node_width = SubElement(node_size, 'width')
                    node_width.text = str(imw)
                    node_height = SubElement(node_size, 'height')
                    node_height.text = str(imh)
                    node_object = SubElement(node_root, 'object')
                    node_trackid = SubElement(node_object, 'trackid')
                    node_trackid.text = str(clas[k][1])
                    node_name = SubElement(node_object, 'name')
                    node_name.text = str(clas[k][2])
                    node_bndbox = SubElement(node_object, 'bndbox')
                    node_xmin = SubElement(node_bndbox, 'xmin')
                    node_xmin.text = str(clas[k][6])
                    node_ymin = SubElement(node_bndbox, 'ymin')
                    node_ymin.text = str(clas[k][7])
                    node_xmax = SubElement(node_bndbox, 'xmax')
                    node_xmax.text = str(clas[k][8])
                    node_ymax = SubElement(node_bndbox, 'ymax')
                    node_ymax.text = str(clas[k][9])
                    node_occluded = SubElement(node_object, 'occluded')
                    node_occluded.text = str(clas[k][4])
                    node_generated = SubElement(node_object, 'generated')
                    node_generated.text = '0'
                    count+=1
                else:
                    node_object = SubElement(node_root, 'object')
                    node_trackid = SubElement(node_object, 'trackid')
                    node_trackid.text = str(clas[k][1])
                    node_name = SubElement(node_object, 'name')
                    node_name.text = str(clas[k][2])
                    node_bndbox = SubElement(node_object, 'bndbox')
                    node_xmin = SubElement(node_bndbox, 'xmin')
                    node_xmin.text = str(clas[k][6])
                    node_ymin = SubElement(node_bndbox, 'ymin')
                    node_ymin.text = str(clas[k][7])
                    node_xmax = SubElement(node_bndbox, 'xmax')
                    node_xmax.text = str(clas[k][8])
                    node_ymax = SubElement(node_bndbox, 'ymax')
                    node_ymax.text = str(clas[k][9])
                    node_occluded = SubElement(node_object, 'occluded')
                    node_occluded.text = str(clas[k][4])
                    node_generated = SubElement(node_object, 'generated')
                    node_generated.text = '0'
            else:
                xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
                img_newxml = os.path.join(xml_path, img_xml)
                file_object = open(img_newxml, 'wb')
                file_object.write(xml)
                file_object.close()
                continue


if __name__ == "__main__":
    #图像文件所在位置
    img_path = r"F:\d\data_tracking_image_2\training\image_02\0000\000000.png"
    #标注文件所在位置
    txt_path=r"F:\d\data_tracking_label_2\training\label_02\0019.txt" #修改
    #txt转化成xml格式后存放的文件夹
    xml_path=r"F:\d\data_tracking_label_2\training\label_02\0019" #修改
    txt_xml(img_path, txt_path, xml_path)
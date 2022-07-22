#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:22:21 2021

@author: headway
"""
import numpy as np
import os, shutil, sys
import random

#GPU 사용시 풀어 놓을 것
"""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""




ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
#--------------------------------------------------
src_dir = os.path.join(ROOT_DIR,'DB','images')
dst_dir = os.path.join(ROOT_DIR,'DB','train')
base_dir = os.path.join(ROOT_DIR,'DB')
ratio = 0.7
option_move = True       #파일을 옮길지 여부
#--------------------------------------------------


#소스 디렉토리가 없으면 만든다.
if not os.path.isdir(src_dir):
        os.mkdir(src_dir)
# 타겟 디렉토리가 없으면 만든다.
if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

total_files = len(os.listdir(src_dir))

SAMPLE_IMG_COUNT = int(total_files*ratio)

print('취득 이미지 갯수: {0} sample 수 : {1}'.format(total_files,SAMPLE_IMG_COUNT))

proc_num = 0

if total_files >  SAMPLE_IMG_COUNT :
    files = os.listdir(src_dir)
    random.shuffle(files)
    sample_files = [ files[i] for i in range(SAMPLE_IMG_COUNT)]
    for file in sample_files:
        
        try:
            
            src = os.path.join(src_dir,file)
            dst = os.path.join(dst_dir,file)

            if option_move :
                shutil.move(src,dst)
            else:
                shutil.copy(src,dst)
            proc_num =  proc_num + 1
            
            print('처리상황: ', proc_num*100/total_files + '%')
        except Exception as e:
            pass
        
    print("처리완료")
else:
      print('취득 이미지 이미지가 충분하지 않습니다.:',total_files)          
        
        
        
        






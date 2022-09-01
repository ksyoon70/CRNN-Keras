# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 19:12:57 2022

@author: headway
"""
#이 prediction 파일은 resnet50을 이용한 문자 인식하는 코드이다.
import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.models import load_model
import time 
from Model_ResNet import get_Model, CTCLayer

# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#---------------------------------------------
ROOT_DIR = os.path.dirname(__file__)
class_str = 'char'
img_width = 224
img_height = 224
batch_size = 32
EPOCHS =  100
MODEL_PATH = 'LSTM_epoch_20220901-163152_val_loss_0.4244.h5'
#WEIGHT_PATH = os.path.join(ROOT_DIR,'trained','LSTM_crnn_20220901-123614_weights_epoch_025_val_loss_0.314.h5')
WEIGHT_PATH = os.path.join(ROOT_DIR,'trained','LSTM_ResNet50_20220901-162838_weights_epoch_015_val_loss_0.377.h5')
label_dir = os.path.join(ROOT_DIR,'DB','train') #여기는 변경하지 않는다.
src_dir = os.path.join(ROOT_DIR,'DB','train')
SHOW_IMAGE = False  #이미지를 보여 줄지여부
#---------------------------------------------

def get_model_path(model_type, backbone="resnet50"):
    """Generating model path from model_type value for save/load model weights.
    inputs:
        model_type = "rpn", "faster_rcnn"
        backbone = "vgg16", "mobilenet_v2"
    outputs:
        model_path = os model path, for example: "trained/rpn_vgg16_model_weights.h5"
    """
    main_path = "trained"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "{}_{}_{}_weights_".format(model_type, backbone,datetime.now().strftime("%Y%m%d-%H%M%S")))
    return model_path

def encode_single_sample(img_path, label):
  # 1. Read image
  img = tf.io.read_file(img_path)
  # 2. Decode and convert to grayscale
  img = tf.io.decode_png(img, channels=3)
  # 3. Convert to float32 in [0, 1] range
  #img = tf.image.convert_image_dtype(img, tf.float32)
  # 4. Resize to the desired size
  img = tf.image.resize(img, [img_height, img_width])
  # 5. Transpose the image because we want the time
  # dimension to correspond to the width of the image.
  img = tf.transpose(img, perm=[1, 0, 2])
  # 6. Map the characters in label to numbers
  label = char_to_num(tf.strings.unicode_split(label, input_encoding='UTF-8'))
  # 7. Return a dict as our model is expecting two inputs
  return {'image': img, 'label': label}



image_ext = ['jpg','JPG','png','PNG']

img_list = [fn for fn in os.listdir(src_dir)
             if any(fn.endswith(ext) for ext in image_ext)]


max_length = 0
imgs = []
labels = []



for filename in img_list:
  imgs.append(os.path.join(src_dir,filename))
  
  basename =os.path.basename(filename)
  label = basename.split('_')[-1]
  label = label[0:-4]
  labels.append(label)

print('이미지갯수 : {}, 검지 레이블 갯수: {}'.format(len(imgs),len(labels)))



label_list = [fn for fn in os.listdir(label_dir)
             if any(fn.endswith(ext) for ext in image_ext)]
gtlabels = []

for filename in label_list:
  basename =os.path.basename(filename)
  gtlabel = basename.split('_')[-1]
  gtlabel = gtlabel[0:-4]
  gtlabels.append(gtlabel)
  if len(gtlabel) > max_length:
    max_length = len(gtlabel)

print('GT 검지 레이블 갯수: {} max label length {}'.format(len(gtlabels), max_length))

characters = set(''.join(gtlabels))
#print(characters)

char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), num_oov_indices=0, mask_token=None, invert=True
)

# print(labels[0])
# encoded = char_to_num(tf.strings.unicode_split(labels[0], input_encoding='UTF-8'))
# print(encoded)

#x_train, x_val, y_train, y_val = train_test_split(imgs, labels, test_size=0, random_state=2021)
x_val = imgs
y_val = labels

#print('train: x_tain: {} y_train: {}'.format(len(x_train), len(y_train)))
print('test: x_val: {} y_val: {}'.format(len(x_val), len(y_val)))


validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)



# Get the model
# model = get_Model(training=False,categories_len=len(char_to_num.get_vocabulary()),img_shape=[img_width,img_height,1])
# model.summary()
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'CTCLayer':CTCLayer})
model.load_weights(WEIGHT_PATH)

prediction_model = keras.models.Model(
  model.get_layer(name='image').input, model.get_layer(name='dense2').output
)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode('utf-8')
        output_text.append(res)
    return output_text


start_time = time.time() # strat time

total_test_files = len(img_list)
recog_count = 0
fail_count = 0
false_recog_count = 0  #오인식 카운트
true_recog_count = 0

for batch in validation_dataset:
    batch_images = batch['image']
    GT_labels = batch['label']
    
    gt_text = []
    for gtlabel in GT_labels:
        res = tf.strings.reduce_join(num_to_char(gtlabel)).numpy().decode('utf-8')
        gt_text.append(res)

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)
    
    batch_size = batch_images.shape[0]
    
    for ix, pred_text in enumerate(pred_texts):
        if pred_text == '[UNK]' :
            fail_count += 1
        else :
            recog_count += 1
            if pred_text == gt_text[ix] :
                true_recog_count += 1
            else :
                false_recog_count += 1
    
    #이미지를 보여 준다.
    if SHOW_IMAGE :
        batch_images_show = batch_images/255
        _, axes = plt.subplots(8, 4, figsize=(16, 12))
    
        for img, text, ax in zip(batch_images_show, pred_texts, axes.flatten()):
            img = img.numpy().squeeze()
            #img = img.T
            img = np.swapaxes(img,0,1)
    
            ax.imshow(img, cmap='gray')
            ax.set_title(text)
            ax.set_axis_off()
            
end_time = time.time()         
print("수행시간: {:.2f}".format(end_time - start_time))
print("건당 수행시간 : {:.2f}".format((end_time - start_time)/total_test_files))             
print('인식률: {:}'.format(recog_count) +'  ({:.2f})'.format(recog_count*100/total_test_files) + ' %')
print('정인식: {:}'.format(true_recog_count) +'  ({:.2f})'.format(true_recog_count*100/recog_count) + ' %')
print('오인식: {:}'.format(false_recog_count) +'  ({:.2f})'.format(false_recog_count*100/recog_count) + ' %')
print('인식실패: {}'.format(fail_count) +'  ({:.2f})'.format(fail_count*100/total_test_files) + ' %')






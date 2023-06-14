# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 19:12:57 2022

@author: headway
"""
#이 train 파일은 resnet50을 이용한 문자 인식하는 코드이다.
from genericpath import isdir
import os,shutil,sys
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
from tensorflow.keras.optimizers import Adadelta

from Model_ResNet import get_Model , get_FineTuneModel
import math
import random 

# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#---------------------------------------------
img_width = 224
img_height = 224
batch_size = 32
EPOCHS =  100
OBJECT_DETECTION_API_PATH = 'C://SPB_Data//RealTimeObjectDetection-main'
class_strings = ['ch','reg']  #ch 는 문자이디ㅏ.reg 는 지역문자이다.
USE_ADADELTA = False  #Adadelta   사용 여부
patience = 10    #업데이트 기다리는 기간
TEST_PAGE_NUM = 5 #테스트 페이지 보여주는 개수
START_LAYER = 120
LAYERS_TRAINABLE = 7 # 트레인 가능한 갯수

#영상이 늘어남에 따라 메모리 부족이 생김. 이에 .shuffle(len(x_val)*2)을 주석 처리함 23년1월29일
#---------------------------------------------

if USE_ADADELTA:
    EPOCHS = 800


def makeGrey3DImage(param):

    img = param['image']
    label = param['label']
    
    param['image'] = tf.image.rgb_to_yiq(img)

    return param

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
    model_path = os.path.join(main_path, "{}_{}_{}__finetune_{}_weights_".format(model_type, backbone,datetime.now().strftime("%Y%m%d-%H%M%S"),tainableLen))
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


ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)



#characters = set(''.join(labels))


categorie_filename = None
categorie_prefix = 'None'
model_dir= None

for class_str in class_strings:
    
    for tainableLen in range(START_LAYER,LAYERS_TRAINABLE+START_LAYER):

        if class_str == 'ch':        #문자 검사
            categorie_filename = 'chcrnn_categories.txt'
            categorie_prefix = 'char'
            model_dir = 'char_crnn_model'
            
        elif class_str == 'reg':    #지역 검사
            categorie_filename = 'regcrnn_categories.txt'
            categorie_prefix = 'reg'
            model_dir = 'reg_crnn_model' 
        else:
            print('카테고리 정의가 없습니다. 종료')
            sys.exit(0)
        
        test_dir = categorie_prefix + '_' + 'train'
        src_dir = os.path.join(ROOT_DIR,'DB', test_dir)
        
        if not os.path.exists(src_dir):
            print('train 디렉토리가 존재하지 않습니다. {}'.format(src_dir))
            sys.exit(0)
            
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
            
            if len(label) > max_length:
                max_length = len(label)
            
        labels = [label.ljust(max_length,' ') for label in labels]
        characters = sorted(list(set([char for label in labels for char in label])))
        
        print(len(imgs), len(labels), max_length)    
            
        with open(categorie_filename, "w", encoding='utf8') as f:
            for categorie in characters :
                f.write(categorie + '\n')
        
        char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=list(characters), num_oov_indices=0, mask_token=None
        )
        
        num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=char_to_num.get_vocabulary(), num_oov_indices=0, mask_token=None, invert=True
        )
        
        # print(labels[0])
        # encoded = char_to_num(tf.strings.unicode_split(labels[0], input_encoding='UTF-8'))
        # print(encoded)
        
        #한글자 한글자를 표시하기위한 테스트 코드
        # for index in range(20) :
        #     preview = encode_single_sample(imgs[index], labels[index])
        #     plt.title(str(preview['label'].numpy()))
        #     plt.imshow(preview['image'].numpy().squeeze())
        #     plt.show()
        
        x_train, x_val, y_train, y_val = train_test_split(imgs, labels, test_size=0.2, random_state=2021)
        
        
        print('train: x_tain: {} y_train: {}'.format(len(x_train), len(y_train)))
        print('test: x_val: {} y_val: {}'.format(len(x_val), len(y_val)))
        
        IMG_SIZE = 225
        
        upper = 45 * (math.pi/180.0) # degrees -> radian
        lower = -45 * (math.pi/180.0)
        
        def rand_degree():
            return random.uniform( lower , upper )
        
        def resize_and_rescale(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
            image = (image / 255.0)
            return image, label
        
        def augment(image_label, seed):
            image, label = image_label
            #image, label = resize_and_rescale(image, label)
            image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
            # Make a new seed
            new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
            # Random crop back to the original size
            image = tf.image.stateless_random_crop(
                image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
            # Random brightness
            image = tf.image.stateless_random_brightness(
                image, max_delta=0.5, seed=new_seed)
            image = tf.image.rotate( image , rand_degree() )
            image = tf.clip_by_value(image, 0, 1)
            return image, label
        
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (
            train_dataset
            #.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(
                encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 
            #.shuffle(len(x_train)*2)
            .shuffle(10 * batch_size)
        )
        
        
        # train_dataset = (
        #     train_dataset.map(
        #        lambda x : makeGrey3DImage(x)
        #     )
        
        # )
        
        
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        
        validation_dataset = (
            validation_dataset.map(
                encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            #.shuffle(len(x_val)*2)
            .shuffle(10 * batch_size)
        )
        
        
        
        # Get the model
        model = get_FineTuneModel(training=True,categories_len=len(char_to_num.get_vocabulary()),img_shape=[img_width,img_height,3],trainableLen = tainableLen)
        
        # Optimizer
        if USE_ADADELTA :
            opt = Adadelta()
        else:
            opt = keras.optimizers.Adam()
        
        # Compile the model and return
        model.compile(optimizer=opt)
        model.summary()
        
        
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        )
        
        model_sub_path_str = get_model_path('LSTM',backbone=categorie_prefix)
        
        weight_filename =  model_sub_path_str + "epoch_{epoch:03d}_val_loss_{val_loss:.3f}.h5"
        checkpoint_callback = ModelCheckpoint(filepath=weight_filename , monitor="val_loss", save_freq='epoch',save_best_only=True, verbose=1, mode='auto' ,save_weights_only=True)
        
        class CustomHistory(tf.keras.callbacks.Callback):
            def init(self, logs={}):
                self.train_loss = []
                self.val_loss = []
                
            def on_epoch_end(self, epoch, logs={}):
                if len(self.val_loss):
                    if logs.get('val_loss') < min(self.val_loss) :
                        global weight_filename
                        weight_filename = model_sub_path_str + "epoch_{:03d}_val_loss_{:.4f}.h5".format(epoch+1,logs.get('val_loss'))
                self.train_loss.append(logs.get('loss'))
                self.val_loss.append(logs.get('val_loss'))
                #print('\nepoch={}, 현재 최대 val_acc={}'.format(epoch,max(self.val_acc)))
        
        
        custom_hist = CustomHistory()
        custom_hist.init()
        
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=EPOCHS,
            callbacks=[early_stopping,checkpoint_callback,custom_hist],
        )
        
        # acc = history.history['acc']
        # val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        
        
        #model.save(model_save_filename,)
        
        
        # plt.plot(epochs, loss, 'bo', label ='Training acc')
        # plt.plot(epochs, val_loss, 'b', label ='Validation acc')
        
        # plt.title('Training and validation accuracy')
        # plt.legend()
        # plt.figure()

        
        Labelstr = 'Training loss {:03d}'.format(tainableLen)
        plt.plot(epochs, loss, 'bo', label = Labelstr)
        Labelstr = 'Validation loss {:03d}'.format(tainableLen)
        plt.plot(epochs, val_loss, 'b', label = Labelstr)
        
        Titlestr = 'Training and validation loss {:03d}'.format(tainableLen)
        plt.title(Titlestr)
        plt.legend()
        plt.figure()
        
        plt.show()
        
        prediction_model = keras.models.Model(
        model.get_layer(name='image').input, model.get_layer(name='dense2').output
        )
        
        # Save model
        model_save_filename = 'LSTM_ResNet_model' + '_' + categorie_prefix + '_'  +"{}_finetune-model_{}_epoch_{}_val_loss_{:.4f}.h5".format(datetime.now().strftime("%Y%m%d-%H%M%S"),tainableLen,len(loss),val_loss[-1])
        prediction_model.save(model_save_filename)
        
        def decode_batch_predictions(pred):
            input_len = np.ones(pred.shape[0]) * pred.shape[1]
            # Use greedy search. For complex tasks, you can use beam search
            results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                :, :max_length
            ]
            decoded = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
            numpy_array = np.asarray(decoded)
            log_probs  = numpy_array[1]
            probabilities = np.exp(-log_probs)  # log probability로 반환하기 때문에 지수 함수로 바꿔줘야 한다.
            # Iterate over the results and get back the text
            output_text = []
            probs = probabilities.reshape((-1)).tolist()
            for ix, res in enumerate(results):
                ch = tf.strings.reduce_join(num_to_char(res)).numpy().decode('utf-8')
                output_text.append(ch)
            return output_text,probs
        
        for batch in validation_dataset.take(TEST_PAGE_NUM):
            batch_images = batch['image']
            preds = prediction_model.predict(batch_images)
            pred_texts, probs = decode_batch_predictions(preds)
            batch_images_show = batch_images/255
            _, axes = plt.subplots(8, 4, figsize=(16, 12))
        
            for img, text, prob, ax in zip(batch_images_show, pred_texts,probs, axes.flatten()):
                img = img.numpy().squeeze()
                #img = img.T
                img = np.swapaxes(img,0,1)
        
                ax.imshow(img, cmap='gray')
                #인식 내용과 확률을 표시한다.
                text_str = '{}  {:.2f}%'.format(text,prob*100)
                ax.set_title(text_str)
                ax.set_axis_off()
                
        #기존 폴더 아래 있는 출력 폴더를 지운다.
        # model_path = os.path.join(OBJECT_DETECTION_API_PATH,model_dir)
        # if not os.path.isdir(model_path) :
        #     os.mkdir(model_path)
            
        # if os.path.exists(model_path):
        #     model_list = os.listdir(model_path)
        #     if len(model_list) :
        #         for fn in model_list:
        #             os.remove(os.path.join(model_path,fn))
                
        # #결과 파일을 복사한다.
        # #weight file 복사
        # src_fn = weight_filename
        # dst_fn = os.path.join(model_path,os.path.basename(src_fn))
        # shutil.copy(src_fn,dst_fn)
        # # 카테고리 파일 복사
        # src_fn = categorie_filename
        # dst_fn = os.path.join(model_path,src_fn)
        # shutil.copy(src_fn,dst_fn)
        # # 모델 파일 복사
        # src_fn = model_save_filename
        # dst_fn = os.path.join(model_path,src_fn)
        # shutil.copy(src_fn,dst_fn)    
            






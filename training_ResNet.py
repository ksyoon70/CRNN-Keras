# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 19:12:57 2022

@author: headway
"""
#이 train 파일은 resnet50을 이용한 문자 인식하는 코드이다.
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

from Model_ResNet import get_Model

# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#---------------------------------------------
class_str = 'char'
img_width = 224
img_height = 224
batch_size = 32
EPOCHS =  100
#---------------------------------------------




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


ROOT_DIR = os.path.dirname(__file__)

src_dir = os.path.join(ROOT_DIR,'DB','train')

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

print(len(imgs), len(labels), max_length)

#characters = set(''.join(labels))
characters = sorted(list(set([char for label in labels for char in label])))
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

#한글자 한글자를 표시하기위한 테스트 코드
# for index in range(20) :
#     preview = encode_single_sample(imgs[index], labels[index])
#     plt.title(str(preview['label'].numpy()))
#     plt.imshow(preview['image'].numpy().squeeze())
#     plt.show()

x_train, x_val, y_train, y_val = train_test_split(imgs, labels, test_size=0.2, random_state=2021)


print('train: x_tain: {} y_train: {}'.format(len(x_train), len(y_train)))
print('test: x_val: {} y_val: {}'.format(len(x_val), len(y_val)))




train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    .shuffle(len(x_train)*2)
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
    .shuffle(len(x_val)*2)
)



# Get the model
model = get_Model(training=True,categories_len=len(char_to_num.get_vocabulary()),img_shape=[img_width,img_height,3])

# Optimizer
opt = keras.optimizers.Adam()
# Compile the model and return
model.compile(optimizer=opt)
model.summary()


early_stopping = EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
)

model_sub_path_str = get_model_path('LSTM',backbone='ResNet50')

weight_filename = model_sub_path_str + "epoch_{epoch:03d}_val_loss_{val_loss:.3f}.h5"
checkpoint_callback = ModelCheckpoint(filepath=weight_filename , monitor="val_loss", save_freq='epoch',save_best_only=True, verbose=1, mode='auto' ,save_weights_only=True)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping,checkpoint_callback],
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

plt.plot(epochs, loss, 'bo', label ='Training loss')
plt.plot(epochs, val_loss, 'b', label ='Validation loss')

plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.show()

prediction_model = keras.models.Model(
  model.get_layer(name='image').input, model.get_layer(name='dense2').output
)

# Save model
model_save_filename = "LSTM_ResNet_epoch_{}_val_loss_{:.4f}.h5".format(datetime.now().strftime("%Y%m%d-%H%M%S"),val_loss[-1])
prediction_model.save(model_save_filename)

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

for batch in validation_dataset.take(2):
    batch_images = batch['image']
    
    

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)
    batch_images_show = batch_images/255
    _, axes = plt.subplots(8, 4, figsize=(16, 12))

    for img, text, ax in zip(batch_images_show, pred_texts, axes.flatten()):
        img = img.numpy().squeeze()
        #img = img.T
        img = np.swapaxes(img,0,1)

        ax.imshow(img, cmap='gray')
        ax.set_title(text)
        ax.set_axis_off()
        
        
# img = makeGrey3DImage(img)
# preview = encode_single_sample(imgs[0], labels[0])
# plt.title(str(preview['label'].numpy()))
# plt.imshow(preview['image'].numpy().squeeze())

# img_array = np.fromfile(os.path.join(src_dir,img_list[0]), np.uint8)
# img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# print('Filename : {}'.format(img_list[0]))
# plt.title(label)
# plt.imshow(img)





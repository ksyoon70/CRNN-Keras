from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from parameter import *

# 이 파일을 resnet50을 기반으로 테스트를 한 것이다.

class CTCLayer(layers.Layer):
    def __init__(self, name=None,**kwargs):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
        super(CTCLayer, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype='int64')

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def get_Model(training,categories_len, img_shape):
    
    img_w = img_shape[0]
    img_h = img_shape[1]
    img_c = img_shape[2]

    input_shape = (img_w, img_h, 3)    # (224, 224, 3)

    image_input = layers.Input(
        shape=(img_w, img_h, 3), name='image'#, dtype='float32'
    )
    
    labels = layers.Input(name='label', shape=(None,), dtype='float32')
    
    base_model = ResNet50(input_tensor=image_input, include_top=False, pooling = 'avg' , input_shape = input_shape, weights = 'imagenet')
    base_model.trainable = True

    for layer in base_model.layers[:143]:    
         layer.trainable = False 
    
    inner = base_model.get_layer(name='conv5_block1_2_relu').output     #전체 번호판을 인식시키기 위해서 출력을 뺌

    # CNN to RNN
    lastFilterLayer = inner.shape[3]
    print('Last Layer depth {}'.format(lastFilterLayer))
    inner = Reshape(target_shape=((inner.shape[1], inner.shape[2]*inner.shape[3])), name='reshape')(inner)
    #inner = layers.Dropout(0.3)(inner)
    inner = Dense(int(lastFilterLayer/2), activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

    inner = layers.Dropout(0.6)(inner)  # drop out 추가
    #inner = Dense(lastFilterLayer, activation='relu', kernel_initializer='he_normal', name='dense1-1')(inner)
    # RNN layer
    # RNNs
    inner = layers.Bidirectional(layers.LSTM(lastFilterLayer, return_sequences=True, dropout=0.25))(inner)
    inner = layers.Bidirectional(layers.LSTM(lastFilterLayer//2, return_sequences=True, dropout=0.25))(inner)


    # transforms RNN output to character activations:
    # Output layer
    y_pred = layers.Dense(
        categories_len + 1, activation='softmax', name='dense2'
    )(inner)
    
     # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name='ctc_loss')(labels, y_pred)

    if training:
        model = Model(inputs=[base_model.input, labels], outputs=output, name='ocr_model_v1'
        )
        return model
    else:
        return Model(inputs=[base_model.input], outputs=y_pred)


def get_FineTuneModel(training,categories_len, img_shape, trainableLen):
    
    img_w = img_shape[0]
    img_h = img_shape[1]
    img_c = img_shape[2]

    input_shape = (img_w, img_h, 3)    # (224, 224, 3)

    image_input = layers.Input(
        shape=(img_w, img_h, 3), name='image'#, dtype='float32'
    )
    
    labels = layers.Input(name='label', shape=(None,), dtype='float32')
    
    base_model = ResNet50(input_tensor=image_input, include_top=False, pooling = 'avg' , input_shape = input_shape, weights = 'imagenet')
    base_model.trainable = True
    
    endLayer = len(base_model.layers)

    for layer in base_model.layers[: - trainableLen]:    
         layer.trainable = False 
    
    inner = base_model.get_layer(name='conv5_block1_2_relu').output     #전체 번호판을 인식시키기 위해서 출력을 뺌

    # CNN to RNN
    lastFilterLayer = inner.shape[3]
    print('Last Layer depth {}'.format(lastFilterLayer))
    inner = Reshape(target_shape=((inner.shape[1], inner.shape[2]*inner.shape[3])), name='reshape')(inner)
    #inner = layers.Dropout(0.3)(inner)
    inner = Dense(int(lastFilterLayer/2), activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

    inner = layers.Dropout(0.6)(inner)  # drop out 추가
    #inner = Dense(lastFilterLayer, activation='relu', kernel_initializer='he_normal', name='dense1-1')(inner)
    # RNN layer
    # RNNs
    inner = layers.Bidirectional(layers.LSTM(lastFilterLayer, return_sequences=True, dropout=0.25))(inner)
    inner = layers.Bidirectional(layers.LSTM(lastFilterLayer//2, return_sequences=True, dropout=0.25))(inner)


    # transforms RNN output to character activations:
    # Output layer
    y_pred = layers.Dense(
        categories_len + 1, activation='softmax', name='dense2'
    )(inner)
    
     # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name='ctc_loss')(labels, y_pred)

    if training:
        model = Model(inputs=[base_model.input, labels], outputs=output, name='ocr_model_v1'
        )
        return model
    else:
        return Model(inputs=[base_model.input], outputs=y_pred)
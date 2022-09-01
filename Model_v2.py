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

    input_shape = (img_w, img_h, img_c)    # (224, 224, 3)

    input_img = layers.Input(
        shape=(img_w, img_h, 1), name='image', dtype='float32'
    )
    
    labels = layers.Input(name='label', shape=(None,), dtype='float32')

    # Convolution layer (VGG)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_img)  # (None, 128, 64, 64)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)
    inner = layers.Dropout(0.3)(inner)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)
    inner = layers.Dropout(0.3)(inner)
    
    # inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 512)
    # inner = BatchNormalization()(inner)
    # inner = Activation('relu')(inner)
    # inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
    # inner = BatchNormalization()(inner)
    # inner = Activation('relu')(inner)
    # inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

    # inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
    # inner = BatchNormalization()(inner)
    # inner = Activation('relu')(inner)

    # CNN to RNN
    lastFilterLayer = inner.shape[3]
    inner = Reshape(target_shape=((inner.shape[1], inner.shape[2]*inner.shape[3])), name='reshape')(inner)
    inner = Dense(lastFilterLayer, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

    #inner = layers.Dropout(0.2)(inner)  # drop out 추가
    # RNN layer
    # RNNs
    #inner = layers.Bidirectional(layers.LSTM(lastFilterLayer, return_sequences=True, dropout=0.25))(inner)
    #inner = layers.Bidirectional(layers.LSTM(lastFilterLayer, return_sequences=True, dropout=0.25))(inner)
    
    inner = layers.Bidirectional(layers.LSTM(lastFilterLayer*2, return_sequences=True))(inner)
    inner = layers.Bidirectional(layers.LSTM(lastFilterLayer, return_sequences=True))(inner)


    # transforms RNN output to character activations:
    # Output layer
    y_pred = layers.Dense(
        categories_len + 1, activation='softmax', name='dense2'
    )(inner)
    
     # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name='ctc_loss')(labels, y_pred)

    if training:
        model = Model(inputs=[input_img, labels], outputs=output, name='ocr_model_v1'
        )
        return model
    else:
        return Model(inputs=[input_img], outputs=y_pred)



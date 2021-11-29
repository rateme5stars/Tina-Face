import tensorflow as tf
from keras import layers
from keras import Model



class Inception(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1x1_1_1 = layers.Conv2D(filters=128, kernel_size=1, strides=1)
        self.conv1x1_1_2 = layers.Conv2D(filters=32, kernel_size=1, strides=1)
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=1, padding='same')
        #
        self.conv1x1_2_1 = layers.Conv2D(filters=64, kernel_size=1, strides=1)
        self.conv3x3 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.conv5x5 = layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same')
        self.conv1x1_2_2 = layers.Conv2D(filters=32, kernel_size=1, strides=1)

    def call(self, x):
        x_1 = self.conv1x1_2_1(x)
        #
        x_2 = self.conv1x1_1_1(x)
        x_2 = self.conv3x3(x_2)
        #
        x_3 = self.conv1x1_1_2(x)
        x_3 = self.conv5x5(x_3)
        #
        x_4 = self.maxpool(x)
        x_4 = self.conv1x1_2_2(x_4)
        #
        return layers.concatenate([x_1, x_2, x_3, x_4], axis=-1)

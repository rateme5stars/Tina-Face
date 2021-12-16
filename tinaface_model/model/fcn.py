import tensorflow as tf
from keras import layers
from keras import Model


class Bridge(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_c5 = layers.Conv2D(filters=256, kernel_size=1, strides=1, name='conv_c5')
        self.conv_c4 = layers.Conv2D(filters=256, kernel_size=1, strides=1, name='conv_c4')
        self.conv_c3 = layers.Conv2D(filters=256, kernel_size=1, strides=1, name='conv_c3')
        self.conv_c2 = layers.Conv2D(filters=256, kernel_size=1, strides=1, name='conv_c2')

    def call(self, resnet_x):
        bridge5 = self.conv_c5(resnet_x[3])
        bridge4 = self.conv_c4(resnet_x[2])
        bridge3 = self.conv_c3(resnet_x[1])
        bridge2 = self.conv_c2(resnet_x[0])
        return [bridge5, bridge4, bridge3, bridge2]


class Downpath(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pad_p5 = layers.ZeroPadding2D(1)
        self.conv_p5 = layers.Conv2D(filters=256, kernel_size=3, strides=1, name='conv_p5')

        self.pad_p4 = layers.ZeroPadding2D(1)
        self.conv_p4 = layers.Conv2D(filters=256, kernel_size=3, strides=1, name='conv_p4')

        self.pad_p3 = layers.ZeroPadding2D(1)
        self.conv_p3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, name='conv_p3')

    def call(self, bridges):
        p5_1 = bridges[0]
        p5 = tf.image.resize(p5_1, [tf.shape(bridges[1])[1], tf.shape(bridges[1])[2]], 
        method='nearest', preserve_aspect_ratio=False, antialias=False, name=None)
        c4p5 = bridges[1] + p5
        c4p5 = self.pad_p5(c4p5)
        p4_1 = self.conv_p5(c4p5)
        #
        p4 = tf.image.resize(p4_1, [tf.shape(bridges[2])[1], tf.shape(bridges[2])[2]], 
        method='nearest', preserve_aspect_ratio=False, antialias=False, name=None)
        c3p4 = bridges[2] + p4
        c3p4 = self.pad_p4(c3p4)
        p3_1 = self.conv_p4(c3p4)
        #
        p3 = tf.image.resize(p3_1, [tf.shape(bridges[3])[1], tf.shape(bridges[3])[2]], 
        method='nearest', preserve_aspect_ratio=False, antialias=False, name=None)
        c2p3 = bridges[3] + p3
        c2p3 = self.pad_p3(c2p3)
        p2_1 = self.conv_p3(c2p3)
        #
        return [p5_1, p4_1, p3_1, p2_1]


class FCN(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bridge = Bridge()
        self.downpath = Downpath()

    def call(self, resnet_x):
        bridges = self.bridge(resnet_x)
        p_block = self.downpath(bridges)
        return p_block


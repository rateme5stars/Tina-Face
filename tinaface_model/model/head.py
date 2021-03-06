import re
from keras import layers
from keras import Model
import tensorflow as tf


class Head(Model):
    def __init__(self, num_level, **kwargs):
        super().__init__(**kwargs)
        self.level = num_level
        self.classification_conv_list = []
        self.regression_conv_list = []
        
        for _ in range(self.level):
            conv1x1 = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='valid')
            self.classification_conv_list.append(conv1x1)

        for _ in range(self.level):
            conv1x1 = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='valid')
            self.regression_conv_list.append(conv1x1)

        self.classification_conv = layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='valid')
        self.regression_conv = layers.Conv2D(filters=12, kernel_size=1, strides=1, padding='valid')
        self.iou_aware_conv = layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='valid')

    def call(self, inception):

        classification_head_tensor, regression_head_tensor = inception, inception
        for i in range(self.level):
            classification_head_tensor = self.classification_conv_list[i](classification_head_tensor)
            #
            regression_head_tensor = self.regression_conv_list[i](regression_head_tensor)
        #
        classification = self.classification_conv(classification_head_tensor)
        regression = self.regression_conv(regression_head_tensor)
        regression = tf.reshape(regression, (-1, regression.shape[1], regression.shape[2], 3, 4))
        iou_aware = self.iou_aware_conv(regression_head_tensor)

        return [classification, regression, iou_aware]

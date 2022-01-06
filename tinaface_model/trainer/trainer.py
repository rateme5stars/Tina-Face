from os import name

import tensorflow as tf
import tensorflow_addons as tfa
from tinaface_model.input_pipeline.pipeline import (InputPipeline,
                                                    apply_sequence)
# NOTE (Nghia): equivalent import, but shorter, hence easier to read
# from tinaface_model.input_pipeline import InputPipeline, apply_sequence

from tinaface_model.model.tinaface import TinaFace
from tinaface_model.trainer.target_assigner import TargetAssigner


class Trainer:
    """
    - init: batch size, learning rate, optimizer, number of max epochs
    - loss functions (check paper)
    - validation metrics (Average Precision)
    - logging (Tensorboard, wandb, ...your choice...)
    - training loop (data, back propagation, stop condition, ...)
    - training step (forward prop, get losses, logging, ...)
    - validation/test loop
    - validation/test step
    """

    def __init__(self, model, train_pipeline, valid_pipeline, batchsize, epochs, loss_weights: list, level_loss_weights: list, **kwargs):
        self.model = model
        self.batchsize = batchsize
        self.epochs = epochs
        self.num_level = model.num_level
        self.target_assigner = TargetAssigner(num_level=4) 

        # for total loss
        self.loss_weights = loss_weights
        self.level_loss_weights = level_loss_weights

        self.train_pipeline = train_pipeline
        self.valid_pipeline = valid_pipeline

        # NOTE (Nghia): where is the learning rate ?
        self.opt = tf.keras.optimizers.Adam()
    
    def classification_loss(self, model_output, target):
        loss = tfa.losses.sigmoid_focal_crossentropy(y_pred=model_output, y_true=target, alpha=0.25, gamma=2, name='classification_loss')
        return loss

    def regression_loss(self, model_output, target):
        loss = tfa.losses.giou_lossy_pred(y_pred=model_output, y_true=target, name='regression_loss')
        return loss

    def iouaware_loss(self, model_output, target):
        loss = tf.keras.losses.categorical_crossentropy(y_pred=model_output, y_true=target, name='iouaware_loss')
        return loss

    def loss_single_level(self, model_output, target):
        c_loss = self.classification_loss(model_output=model_output[0], target=target['classification'])
        r_loss = self.regression_loss(model_output=model_output[1], target=target['regression'])
        i_loss = self.iouaware_loss(model_output=model_output[2], target=target['iouaware'])

        total_loss = self.loss_weights[0] * c_loss + self.loss_weights[1] * r_loss + self.loss_weights[2] * i_loss
        return total_loss
    
    def get_total_loss(self, model_output, target):
        total_loss = 0
        for lv in range(self.num_level): 
            # NOTE (Nghia): attention, model_output is a list of tensors for each level
            # target is also a dict of multiple levels
            # self.loss_single_level should digest only a tensor for a clear and simple software design
            total_loss += self.loss_single_level(model_output, target) * self.level_loss_weights[lv]
        return total_loss
        
    def valid(self):
        pass
        output_valid_pipeline = ... # batchsize = 1
        for image, padded_boxes, targets in output_valid_pipeline:
            # image (1, H, W, 3)
            model_output = self.model(image)
            total_loss = ...
            predicted_boxes = ...
            metrics = ... # AP
            # log 10 images only in valid set (optional)
        # log average metrics and losses
        self.log()

    def log(self):
        pass

    def train(self):
        output_train_pipeline = self.train_pipeline.get_tf_dataset().batch(self.batchsize)
        test_data = output_train_pipeline.take(10)

        for _ in range(self.epochs): # recommend: use tqdm for visualizing process
            # NOTE (Nghia): tf dataset should ended with .prefetch()
            for batch_images, _, targets, in test_data.shuffle(buffer_size=50).prefetch(buffer_size=4):
                with tf.GradientTape() as tape:
                    # forward prop
                    model_output = self.model(batch_images)
                    # get loss
                    total_loss = self.get_total_loss(model_output=model_output, target=targets)
                    # back prop
                    gradients = tape.get_gradient(total_loss, tape.watched_variables())
                    self.opt.apply_gradients(zip(gradients, tape.watched_variables()))
                # logging
                # self.log()
            # validation loop and get validation metrics
            # self.valid()
            # save best model based on valid losses/metrics
            # early stopping on valid loss and valid metric (optional)


if __name__ == '__main__':
    model = TinaFace(num_level=4)
    target_assigner = TargetAssigner(num_level=4)

    sequences = [apply_sequence(apply_augmentation=True), apply_sequence(apply_augmentation=False)]

    train_pipeline = InputPipeline(target_assigner=target_assigner,
                                   annotation_dir='/Users/dzungngo/Desktop/Tina_Face/data/imageJSON', 
                                   image_shape=(640, 640),
                                   pre_processing=sequences[1],
                                   augmentation=sequences[0])

    valid_pipeline = InputPipeline(target_assigner=target_assigner,
                                   annotation_dir='/Users/dzungngo/Desktop/Tina_Face/data/valJSON',
                                   image_shape=(640, 640),
                                   pre_processing=sequences[0])

    batch_size = 32
    epochs = 10
    loss_weights = tf.constant([1, 1, 1], dtype=tf.float32)
    level_loss_weights = tf.constant([1, 1, 1, 1], dtype=tf.float32)

    trainer = Trainer(model=model, 
                      train_pipeline=train_pipeline, 
                      valid_pipeline=valid_pipeline, 
                      batchsize=batch_size, 
                      epochs=epochs,
                      loss_weights=loss_weights,
                      level_loss_weights=level_loss_weights)


    trainer.train()

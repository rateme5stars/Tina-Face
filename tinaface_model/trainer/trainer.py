import tensorflow as tf
import tensorflow_addons as tfa
from tinaface_model.input_pipeline import (InputPipeline, apply_sequence)
import matplotlib.pyplot as plt

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

    def __init__(self, model, lr, batchsize, epochs, train_pipeline, valid_pipeline, loss_weights: list, level_loss_weights: list, **kwargs):
        self.model = model
        self.batchsize = batchsize
        self.epochs = epochs
        self.num_level = model.num_level
        self.target_assigner = TargetAssigner(num_level=model.num_level) 

        # for total loss
        self.loss_weights = loss_weights
        self.level_loss_weights = level_loss_weights

        self.train_pipeline = train_pipeline
        self.valid_pipeline = valid_pipeline

        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)
        # NOTE (Nghia): from_logits=True because there is no activation in classification head
        self.focal_loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2, from_logits=True)
        self.g_iou_loss = tfa.losses.GIoULoss()
        # NOTE (Nghia): use BinaryCrossentropy instead of CategoricalCrossentropy
        # because CategoricalCrossentropy expects the target to be one-hot
        # which is not in our case
        # here we don't use from_logits=True because there is already sigmoid activation
        # in iou aware head.
        # I recommend to remove this activation this head.py,
        # so that the output of iou aware head is logits and coherent with classification head
        self.crossentropy_loss = tf.keras.losses.BinaryCrossentropy()
    
    def classification_loss(self, model_output, target):
        loss = self.focal_loss(y_pred=model_output, y_true=target)
        return tf.reduce_mean(loss)

    # NOTE (Nghia):
    # in box regression, we should combine L1 loss and GIoU loss
    # At the begin, GIoU loss is difficult to converge due to vanishing gradient. 
    # The GIoU loss usually works well after a certain number of train iterations.
    # That's why in Tina Face paper, the CIoU or DIoU loss were used instead of GIoU
    # If we use CIoU or DIoU loss, maybe L1 loss isn't necessary
    
    # NOTE (Nghia): IMPORTANT
    # bounding box target should be normalized by image size
    # so that the model would learn faster 
    def regression_loss(self, model_output, target):
        b = target.shape[0]
        model_output = tf.reshape(model_output, (b, -1, 3, 4))
        target = tf.reshape(target, (b, -1, 3, 4))
        non_null_target_filter = tf.reduce_max(target, -1) > 0
        model_output = tf.boolean_mask(model_output, non_null_target_filter)
        target = tf.boolean_mask(target, non_null_target_filter)
        gloss = self.g_iou_loss(y_pred=model_output, y_true=target)
        if tf.size(target) == 0:
            l1loss = 0
        else:
            l1loss =  tf.keras.metrics.mean_absolute_error(target, model_output)
            l1loss = tf.reduce_mean(l1loss)
        return gloss + l1loss

    def iouaware_loss(self, model_output, target):
        loss = self.crossentropy_loss(y_pred=model_output, y_true=target)
        return tf.reduce_mean(loss)

    def loss_in_single_level(self, model_output, target, lv):
        c_loss = self.classification_loss(model_output=model_output[lv][0], target=target[lv]['classification'])
        r_loss = self.regression_loss(model_output=model_output[lv][1]/640, target=target[lv]['regression']/640)
        i_loss = self.iouaware_loss(model_output=model_output[lv][2], target=target[lv]['iouaware'])
        single_level_loss = self.loss_weights[0] * c_loss + self.loss_weights[1] * r_loss + self.loss_weights[2] * i_loss
        return single_level_loss
    
    def get_total_loss(self, model_output, target):
        total_loss = 0
        for lv in range(self.num_level): 
            # NOTE (Nghia): attention, model_output is a list of tensors for each level
            # target is also a dict of multiple levels
            # self.loss_single_level should digest only a tensor for a clear and simple software design
            total_loss += self.loss_in_single_level(model_output, target, lv) * self.level_loss_weights[lv]
        return total_loss

    def train(self):
        output_train_pipeline = self.train_pipeline.get_tf_dataset().batch(self.batchsize)
        test_data = output_train_pipeline.take(1)
        train_losses = []
        for e in range(self.epochs): # recommend: use tqdm for visualizing process
            print(f'epoch: {e}')
            # NOTE (Nghia): tf dataset should ended with .prefetch()
            for batch_images, _, targets, in test_data.shuffle(buffer_size=50).prefetch(buffer_size=4):
                with tf.GradientTape() as tape:
                    # forward prop
                    model_output = self.model(batch_images)
                    # get loss
                    total_train_loss = self.get_total_loss(model_output=model_output, target=targets)
                    train_losses.append(total_train_loss)
                    print(float(total_train_loss))
                    # back prop
                    gradients = tape.gradient(total_train_loss, tape.watched_variables())
                    self.opt.apply_gradients(zip(gradients, tape.watched_variables()))
                # logging
                # self.log()
            # validation loop and get validation metrics
            self.valid(alpha=0.5, threshold=0.6)
            # save best model based on valid losses/metrics
            # early stopping on valid loss and valid metric (optional)
        plt.plot(train_losses)
    
    def valid(self, alpha, threshold):
        output_valid_pipeline = self.valid_pipeline.get_tf_dataset().batch(1) # batchsize = 1
        test_valid_data = output_valid_pipeline.take(1)
        for images, padded_boxes, targets in test_valid_data.prefetch(buffer_size=1):
            valid_output = self.model(images)
            total_valid_loss = self.get_total_loss(model_output=valid_output, target=targets)

            predicted_boxes = []
            for out in valid_output:
                score = pow(out[0], alpha) * pow(out[2], 1-alpha)
                condition = score > threshold
                prediction = tf.boolean_mask(out[1], condition)
                if tf.shape(prediction)[0] != 0:
                    predicted_boxes.append(prediction)
        
            predicted_boxes = tf.concat(predicted_boxes, axis=0)
            
            num_boxes = tf.cast(padded_boxes[0][0][0], tf.int32)
            labels = padded_boxes[1:num_boxes + 1, :]
            
            metrics = self.calculate_mAP(labels=labels, predictions=predicted_boxes)
        self.log()
    
    def calculate_mAP(self, labels, predictions):
        pass

    def log(self):
        print("Logging")



if __name__ == '__main__':
    model = TinaFace(num_level=4)
    target_assigner = TargetAssigner(num_level=4)

    sequences = [apply_sequence(apply_augmentation=True), apply_sequence(apply_augmentation=False)]

    train_pipeline = InputPipeline(target_assigner=target_assigner,
                                   annotation_dir='data/imageJSON', 
                                   image_shape=(640, 640),
                                   pre_processing=sequences[1],
                                   augmentation=sequences[0])

    valid_pipeline = InputPipeline(target_assigner=target_assigner,
                                   annotation_dir='data/valJSON',
                                   image_shape=(640, 640),
                                   pre_processing=sequences[1])

    batch_size = 1
    epochs = 1
    lr = 0.0001
    # loss_weights = tf.constant([1, 1, 1], dtype=tf.float32)
    loss_weights = tf.constant([1, 1, 1], dtype=tf.float32)
    level_loss_weights = tf.constant([1, 1, 1, 1], dtype=tf.float32)

    trainer = Trainer(model=model,
                      lr=lr, 
                      batchsize=batch_size,
                      epochs=epochs,
                      train_pipeline=train_pipeline, 
                      valid_pipeline=valid_pipeline, 
                      loss_weights=loss_weights,
                      level_loss_weights=level_loss_weights)


    trainer.train()

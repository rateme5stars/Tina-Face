from os import name
import tensorflow as tf
import tensorflow_addons as tfa

from tinaface_model.input_pipeline.pipeline import InputPipeline
from tinaface_model.trainer.target_assigner import TargetAssigner
from tinaface_model.model.tinaface import TinaFace


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

    def loss_a_level(self):
        c_loss = self.classification_loss(..., ...)
        r_loss = self.regression_loss(..., ...)
        i_loss = self.iouaware_loss(..., ...)

        total_loss = self.w1 * c_loss + self.w2 * r_loss + self.w3 * i_loss
        return total_loss
    
    def get_total_loss(self, model_outputs, targets):
        total_loss = 0
        for lv in range(self.num_level): 
            total_loss += self.loss_a_level(model_outputs) * self.level_loss_weights[lv]
        return total_loss
        
    def valid(self):
        pass

    def log(self):
        pass

    def train(self):
        for epoch in range(...): # recommend: use tqdm for visualizing process

            # for batch_images, bbox in input_pipeline.shuffle().batch():

                # targets = self.target_assigner(bbox) # check if it can work with batch
                with tf.GradientTape() as tape:
                    pass
                    # forward prop
                    model_outputs = self.model(batch_images)
                    # get loss
                    total_loss = self.get_total_loss(model_outputs, targets)
                    # back prop
                    grads = tape.get_gradient(total_loss)
                    optimizer.optimize(grad, ...)
                # logging
                self.log()
            # validation loop and get validation metrics
            self.valid()
            # early stopping on valid loss and valid metric


            


if __name__ == '__main__':
    pass



"""
- write the whole training loop
- factorize codes by sections (loss_functions, calculate losses, forward prop, backward prop)
- identify hyperparameters in training loop, put them to init
- write validation loop
- add logging
"""
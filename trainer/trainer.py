import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


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

    def __init__(self, batchsize, epochs, alpha, gamma, w1, w2, w3,  **kwargs):
        self.batchsize = batchsize
        self.epochs = epochs

        # for classification loss
        self.alpha = alpha
        self.gamma = gamma

        # for total loss
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
    
    def classification_loss(self):
        pass

    def regression_loss(self):
        pass

    def iouaware_loss(self):
        pass

    
    def total_loss(self):
        pass



"""
- write the whole training loop
- factorize codes by sections (loss_functions, calculate losses, forward prop, backward prop)
- identify hyperparameters in training loop, put them to init
- write validation loop
- add logging
"""
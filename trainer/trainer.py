import tensorflow as tf
from tensorflow import keras

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

    def __init__(self, **kwargs):
        pass
         


"""
The losses of classification, regression and IoU
prediction are focal loss, DIoU loss and cross-entropy loss,
respectively.
"""


"""
- write the whole training loop
- factorize codes by sections (loss_functions, calculate losses, forward prop, backward prop)
- identify hyperparameters in training loop, put them to init
- write validation loop
- add logging
"""
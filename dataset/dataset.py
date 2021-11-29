from typing import Dict, List
from imgaug.augmenters.size import Resize
import tensorflow as tf
import numpy as np
import os
import json 

import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def np_boxes_to_imgaug_boxes(boxes: np.ndarray, image_shape: tuple) -> BoundingBoxesOnImage:
    return BoundingBoxesOnImage([
        BoundingBox(*list(box))
        for box in boxes
    ], shape=image_shape)

class InternetFaces():
    def __init__(self, annotation_dir: str, image_shape: tuple, augmentation_seq: iaa.Sequential = None):
        """...
        """
        self.annotation_dir = annotation_dir
        self.json_image_dict = self.get_json_image_dict()
        self.augmentation_seq = augmentation_seq
        self.H, self.W = image_shape

    def get_json_image_dict(self) -> Dict[str, str]:
        """
        Returns
        -------
        Dict of:
            - key: path to json annotation
            - value: path to image
        """
        # json_names = []
        # for name in os.listdir(self.annotation_dir):
        #     if name.endswith(...):
        #         path = os.path.join(..., ...)
        #         json_names.append(path)
        json_image_dict = {}

        json_paths = [
            os.path.join(self.annotation_dir, name)
            for name in os.listdir(self.annotation_dir)
            if name.endswith(".json")]

        for json_path in json_paths:
            with open(json_path) as f:
                annotation = json.load(f)
                json_image_dict[json_path] = os.path.join(
                    self.annotation_dir, annotation["imagePath"])

        return json_image_dict

    def get_bbox_from_json(self, json_path) -> np.ndarray:
        """Given json path, return bbox with absolute coordinates and format xyxy

        Returns
        -------
        np.array
            float32, shape(n, 4) with n number of boxes
        """
        bboxes = []
        with open(json_path) as f:
            data = json.load(f)
        for annotation in data["shapes"]:
            bbox = annotation["points"][0] + annotation["points"][1]
            bboxes.append(bbox)
        return np.array(bboxes, dtype=np.float32)

    def _generator(self):
        """
        A generator function to yield 2 tensors
        - image: np.ndarray
            float32, shape (H, W, 3)
        - bbox: np.ndarray, absolute coordinate, format xyxy
            float32, shape (100, 4), bbox[0, 0] indicates number of boxes
        """
        # For loop each image path
        for json_path, image_path in self.json_image_dict.items():
            # Load image to tensor
            image_pil = tf.keras.utils.load_img(image_path)
            image = tf.keras.utils.img_to_array(image_pil)
            # Load bbox from to tensor
            bboxes = self.get_bbox_from_json(json_path)
            # Data augmentation
            bbs = np_boxes_to_imgaug_boxes(bboxes, image.shape)
            image_aug, bbs_aug = self.augmentation_seq(
                image=image, bounding_boxes=bbs)
            bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
            bboxes_aug = bbs_aug.to_xyxy_array()
            # Zero padding for bbox array
            padded_bboxes = np.zeros((100, 4), dtype=bboxes_aug.dtype)
            padded_bboxes[0, 0] = bboxes_aug.shape[0]
            padded_bboxes[1:bboxes_aug.shape[0]+1] = bboxes_aug
            # Yield image, bbox
            yield image_aug, padded_bboxes

    def get_tf_dataset(self):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(self.H, self.W, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(100, 4), dtype=tf.float32))
        )


from typing import Dict
import tensorflow as tf
import numpy as np
import os
import json 
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def np_bboxes_to_imgaug_boxes(boxes: np.ndarray, image_shape: tuple) -> BoundingBoxesOnImage:
    return BoundingBoxesOnImage([
        BoundingBox(*list(box))
        for box in boxes
    ], shape=image_shape)
 
def apply_augmentation(rotate): 
    if rotate:
        aug =  iaa.Sequential([
            iaa.Rotate((-10, 10)),
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
            iaa.MultiplyHue((0.5, 1.5))
        ])
    else:
        aug =  iaa.Sequential([
            iaa.Resize({"shorter-side": 640, "longer-side": "keep-aspect-ratio"}),
            iaa.CenterCropToFixedSize(640, 640),
            ])
    return aug

class InputPipeline:
    def __init__(self, annotation_dir: str, image_shape: tuple, augmentation_seq: iaa.Sequential=None, rotate: iaa.Sequential=None):
        """
        Parameter
        ---------
        annotation_dir: Absolute direction to annotation folder
        image_shape: size of output image (640x640 according to paper)
        augmentation_seq: sequential of applied augmentation methods
        """
        self.annotation_dir = annotation_dir
        self.H, self.W = image_shape
        self.json_image_dict = self.get_json_image_dict()
        self.augmentation_seq = augmentation_seq
        self.rotate = rotate
        

    def get_json_image_dict(self) -> Dict[str, str]:
        """
        Returns
        -------
        Dict of:
            - key: path to json annotation
            - value: path to corresponding image
        """
        # json_paths = []
        # for name in os.listdir(self.annotation_dir):
        #     if name.endswith(...):
        #         path = os.path.join(..., ...)
        #         json_paths.append(path)
        
        # 5 lines below are equivalent to 5 lines above
        # json_paths is a list contains paths lead to all annotation json files of each image
        json_paths = [ 
            os.path.join(self.annotation_dir, name) 
            for name in os.listdir(self.annotation_dir)
            if name.endswith(".json")
        ]

        json_image_dict = {}
        for json_path in json_paths:
            with open(json_path) as file:
                annotation = json.load(file)
                json_image_dict[json_path] = os.path.join(self.annotation_dir, annotation["imagePath"])

        return json_image_dict


    def get_bbox_from_json(self, json_path) -> np.ndarray:
        """
        Parameters
        ----------
        Path to annotation of image

        Returns
        -------
        np.array
            float32, shape(n, 4) with n number of boxes
        """
        bboxes = []
        with open(json_path) as f:
            annotation = json.load(f)
        for shape in annotation["shapes"]:
            bbox = shape["points"][0] + shape["points"][1]
            bboxes.append(bbox)
        return np.array(bboxes, dtype=np.float32)
    
    def generator(self):
        """
        Yield 2 tensors
        - image: np.ndarray, float32, shape (H, W, 3)
        - bbox: np.ndarray, absolute coordinate, format xyxy, float32, shape (100, 4), bbox[0, 0] indicates number of boxes
        """
        # For loop each image path
        for json_path, image_path in self.json_image_dict.items():
            # Load image to tensor
            image_pil = tf.keras.utils.load_img(image_path)
            image_arr = tf.keras.utils.img_to_array(image_pil)

            # load bbox to tensor
            bboxes = self.get_bbox_from_json(json_path)

            # data augmentation
            bbs = np_bboxes_to_imgaug_boxes(bboxes, image_arr.shape)
            image_aug, bbs_aug = self.augmentation_seq(image=image_arr, bounding_boxes=bbs)
            bbs_aug = bbs_aug.remove_out_of_image()

            # handle outside bboxes
            bboxes_aug = bbs_aug.to_xyxy_array()
            center_bboxes_aug = (bboxes_aug[:, :2] + bboxes_aug[:, 2:]) / 2 # (n, 2)
            filter_condition = ((center_bboxes_aug > 0).all(axis=-1) & (center_bboxes_aug < 640).all(-1))
            bboxes_aug = bboxes_aug[filter_condition]  
            

            if self.rotate != None:
                bbs_2 = np_bboxes_to_imgaug_boxes(bboxes_aug, image_aug.shape)
                image_aug = tf.cast(image_aug, tf.uint8).numpy()
                image_aug, bbs_aug = self.rotate(image=image_aug, bounding_boxes=bbs_2)
                image_aug = tf.cast(image_aug, tf.float32)
                bboxes_aug = bbs_aug.to_xyxy_array()

            # create padded bboxes
            padded_bboxes = np.zeros((100, 4), dtype=bboxes_aug.dtype)
            padded_bboxes[0, 0] = bboxes_aug.shape[0]
            padded_bboxes[1 : bboxes_aug.shape[0] + 1] = bboxes_aug
            yield image_aug, padded_bboxes

    def get_tf_dataset(self):
        return tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=(self.H, self.W, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(100, 4), dtype=tf.float32))
        )
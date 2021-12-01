from typing import Dict
import tensorflow as tf
import numpy as np
import os
import json 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def np_bboxes_to_imgaug_boxes(boxes: np.ndarray, image_shape: tuple) -> BoundingBoxesOnImage:
    return BoundingBoxesOnImage([
        BoundingBox(*list(box))
        for box in boxes
    ], shape=image_shape)

class InputPipeline:
    def __init__(self, annotation_dir: str, image_shape: tuple, augmentation_seq: iaa.Sequential = None):
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
            bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
            bboxes_aug = bbs_aug.to_xyxy_array()
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
             
# test
# if __name__ == "__main__":
#     H = W = 640
#     seq = iaa.Sequential([
#         iaa.Resize({"shorter-side": 640, "longer-side": "keep-aspect-ratio"}),
#         iaa.CenterCropToFixedSize(W, H),
#         iaa.Rotate((-10, 10))
#     ])

#     input_pipeline = InputPipeline('/Users/dzungngo/Desktop/Tina_Face/data/testJSON', (640, 640), seq)
#     tf_dataset = input_pipeline.get_tf_dataset()
#     for image_tensor, bboxes_tensor in tf_dataset:
#         image = tf.keras.utils.array_to_img(image_tensor)
        
#         fig, ax = plt.subplots(1)
#         ax.imshow(image)
#         for coor in bboxes_tensor:
#             rect = patches.Rectangle((coor[0], coor[1]), coor[2] - coor[0], coor[3] - coor[1], linewidth=1, edgecolor='r', facecolor="none")
#             ax.add_patch(rect)
#         plt.show()
#         print(bboxes_tensor)
    

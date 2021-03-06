from typing import Dict
import tensorflow as tf
import numpy as np
import os
import json 
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tinaface_model.trainer.target_assigner import TargetAssigner

def np_bboxes_to_imgaug_boxes(boxes: np.ndarray, image_shape: tuple) -> BoundingBoxesOnImage:
    return BoundingBoxesOnImage([
        BoundingBox(*list(box))
        for box in boxes
    ], shape=image_shape)

def apply_sequence(apply_augmentation: bool): 
    if apply_augmentation:
        sequence =  iaa.Sequential([
            iaa.Rotate((-10, 10)),
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
            iaa.MultiplyHue((0.5, 1.5))
        ])
    else:
        sequence =  iaa.Sequential([
            iaa.Resize({"shorter-side": 640, "longer-side": "keep-aspect-ratio"}),
            iaa.CenterCropToFixedSize(640, 640),
            ])
    return sequence

class InputPipeline:
    def __init__(self, target_assigner: TargetAssigner, annotation_dir: str, image_shape: tuple, pre_processing: iaa.Sequential=None, augmentation: iaa.Sequential=None):
        """
        Parameter
        ---------
        annotation_dir: Absolute direction to annotation folder
        image_shape: size of output image (640x640 according to paper)
        pre_processing: sequential of applied pre-processing methods
        augmentation: sequential of applied augmentation methods
        """
        self.annotation_dir = annotation_dir
        self.H, self.W = image_shape
        self.json_image_dict = self.get_json_image_dict()
        self.pre_processing = pre_processing
        self.augmentation = augmentation
        self.target_assigner = target_assigner


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

            # image pre-processing
            bbs = np_bboxes_to_imgaug_boxes(bboxes, image_arr.shape)
            image_aug, bbs_aug = self.pre_processing(image=image_arr, bounding_boxes=bbs)
            bbs_aug = bbs_aug.remove_out_of_image()

            # apply augmentation
            # if self.augmentation != None:
            # NOTE (Nghia): for None comparison, always use "is None" for "is not None"
            # Why? Faster, safer than "==" and "!="
            # http://jaredgrubb.blogspot.com/2009/04/python-is-none-vs-none.html
            if self.augmentation is not None:
                # bbs_2 = np_bboxes_to_imgaug_boxes(bboxes_aug, image_aug.shape)
                image_aug = tf.cast(image_aug, tf.uint8).numpy()
                image_aug, bbs_aug = self.augmentation(image=image_aug, bounding_boxes=bbs_aug)
                image_aug = tf.cast(image_aug, tf.float32)
                bboxes_aug = bbs_aug.to_xyxy_array()

            # handle outside bboxes
            bboxes_aug = bbs_aug.to_xyxy_array()
            center_bboxes_aug = (bboxes_aug[:, :2] + bboxes_aug[:, 2:]) / 2 # (n, 2)
            filter_condition = ((center_bboxes_aug > 0).all(axis=-1) & (center_bboxes_aug < 640).all(-1))
            bboxes_aug = bboxes_aug[filter_condition]  
            
            # create padded bboxes
            padded_bboxes = np.zeros((100, 4), dtype=bboxes_aug.dtype)
            padded_bboxes[0, 0] = bboxes_aug.shape[0]
            padded_bboxes[1 : bboxes_aug.shape[0] + 1] = bboxes_aug
            targets = self.target_assigner.get_target(padded_bboxes)

            yield image_aug, padded_bboxes, targets

    def get_tf_dataset(self):
        ratio = [1, 0.5, pow(0.5, 2), pow(0.5, 3)]
        size = 160
        return tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=(self.H, self.W, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(100, 4), dtype=tf.float32),
                tuple({
                    "classification": tf.TensorSpec(shape=(int(size*ratio[i]), int(size*ratio[i]), 3), dtype=tf.float32),
                    "regression": tf.TensorSpec(shape=(int(size*ratio[i]), int(size*ratio[i]), 3, 4), dtype=tf.float32),
                    "iouaware": tf.TensorSpec(shape=(int(size*ratio[i]), int(size*ratio[i]), 3), dtype=tf.float32)
                } for i in range(4)) 
            )
        )

# if __name__ == "__main__":
#     train_dir = "D:\\Work\\dataset\\internet_faces\\ImageJSON"
#     val_dir = "D:\\Work\\dataset\\internet_faces\\ValJSON"
#     sequences = [apply_sequence(apply_augmentation=True), apply_sequence(apply_augmentation=False)]
#     target_assigner = TargetAssigner(num_level=4)

#     train_pipeline = InputPipeline(target_assigner=target_assigner,
#                                    annotation_dir=train_dir, 
#                                    image_shape=(640, 640),
#                                    pre_processing=apply_sequence(apply_augmentation=False),
#                                    augmentation=sequences[0])
#     # NOTE (Nghia): this line should always ends with .prefetch()
#     dataset = train_pipeline.get_tf_dataset().batch(4).shuffle(512).prefetch(32)
#     for images, bboxes, target in dataset:
#         print(images.shape)
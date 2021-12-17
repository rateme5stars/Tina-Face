import numpy as np
import tensorflow as tf

class TargetAssigner:
    def __init__(self, num_level):
        self.level = num_level 
        self.iou_min = 0.35
        
        self.level_dict = {}
        base_size = 160
        base_anchor_size = [4, 8, 16, 32]
        anchor_size_scales = np.array([4/3, 5/3, 6/3])

        levels = range(4)

        for i in levels:
            size = base_size//pow(2, i)
            self.level_dict[i + 2] = [size, 640//size, base_anchor_size[i]*pow(2, anchor_size_scales)]

        self.anchor_coordinate_all_level = []
        for level in range(num_level):
            level = level + 2
            W = H = self.level_dict[level][0]
            X, Y = tf.meshgrid(tf.range(H), tf.range(W))
            index_tensor = tf.stack([X, Y], axis=-1) # shape(160, 160, 2)
            index_tensor = tf.concat([index_tensor, index_tensor], axis=-1) # shape(160, 160, 4) 
            index_tensor = tf.reshape(index_tensor, (H, W, 1, 4)) # (160, 160, 1, 4)
            index_tensor = tf.repeat(index_tensor, repeats=3, axis=2) # shape(160, 160, 1, 4)
            self.anchor_coordinate = tf.cast(index_tensor, tf.float32)
            init_coor = np.array([[-size/2, -(size*1.3)/2, size/2, (size*1.3)/2] for size in self.level_dict[level][2]]) + self.level_dict[level][1]/2
            init_coor = tf.convert_to_tensor(init_coor, dtype=tf.float32)
            self.anchor_coordinate = init_coor + self.anchor_coordinate * self.level_dict[level][1]
            self.anchor_coordinate_all_level.append(self.anchor_coordinate)
            # anchor_coordinate_all_level = [(160, 160, 3, 4), (80, 80, 3, 4), (40, 40, 3, 4), (20, 20, 3, 4)]
        
    def find_associated_anchors(self, bboxes_coor, level):
        '''
        Find all associated anchors with bbox by bbox indexes on 1 level
        Parameters
        ----------
        bboxes_coor: tf.Tensor, shape(n, 4) with 'n' is the number of bbox in an image
        level: int, stand for level of head
        Return
        ------
        A tensor of all anchor coordinates which associate with each bbox in image
        '''
        topleft_coor = bboxes_coor[:, :2] # (n, 2)
        botright_coor = bboxes_coor[:, 2:] # (n, 2)
        center_bboxes_coor = (botright_coor+topleft_coor)/2 # (n, 2)
        anchor_indexes = tf.math.floor(center_bboxes_coor/(self.level_dict[level+2][1])) # (n, 2)
        anchor_indexes = tf.cast(anchor_indexes, tf.int32) # (n, 2)
        X, Y = anchor_indexes[:, 0], anchor_indexes[:, -1] # (n, ) , (n, )
        anchor_indexes = tf.stack([Y, X], axis=-1) # (n, 2)
        anchors_coor = tf.gather_nd(self.anchor_coordinate_all_level[level], anchor_indexes) # (n, 3, 4)
        return [anchors_coor, anchor_indexes]

    def calculate_areas(self, bboxes_coor, anchors_coor):
        '''
        Calculate bbox areas, anchor areas and their intersect areas on 1 level

        Parameters
        ----------
        bboxes_coor: tf.Tensor, shape(n, 4) with 'n' is the number of bbox in an image
        anchors_coor: tf.Tensor, shape(n, 3, 4) with 'n' is the number of bbox in an image

        Return
        ------
        A list includes bboxe areas, anchor areas and their inter areas
        '''
        topleft_bboxes_coor = bboxes_coor[:, :, :2] # (n, 3, 2)
        botright_bboxes_coor = bboxes_coor[:, :, 2:] # (n, 3, 2)
        bboxes_size = botright_bboxes_coor - topleft_bboxes_coor # (n, 3, 2)
        bboxes_width = bboxes_size[:, :, 0] # (9, 3)
        bboxes_height = bboxes_size[:, :, -1] # (9, 3)
        bboxes_area = bboxes_width * bboxes_height # bboxes area of all bboxes in image with 3 anchor sizes (9, 3)

        topleft_anchors_coor = anchors_coor[:, :, :2] # (n, 3, 2)
        botright_anchors_coor = anchors_coor[:, :, 2:] # (n, 3, 2)
        anchors_size = botright_anchors_coor - topleft_anchors_coor # (n, 3, 2)
        anchors_width = anchors_size[:, :, 0] # (9, 3)
        anchors_height = anchors_size[:, :, -1] # (9, 3)
        anchors_area = anchors_width * anchors_height # bboxes area of all bboxes in image with 3 anchor sizes (9, 3)

        topleft_inter_coor = tf.maximum(topleft_anchors_coor, topleft_bboxes_coor) # (n, 3, 2)
        botright_inter_coor = tf.minimum(botright_anchors_coor, botright_bboxes_coor) # (n, 3, 2)
        inter_size = botright_inter_coor - topleft_inter_coor 
        inter_size = tf.clip_by_value(inter_size, 0, 640) # handle non-intersection case (inter_size < 0)
        inter_width = inter_size[:, :, 0] # (9, 3)
        inter_height = inter_size[:, :, -1] # (9, 3)
        inter_area = inter_width * inter_height # (9, 3)

        return [bboxes_area, anchors_area, inter_area]

    def calculate_iou(self, bboxes_coor):
        '''
        Calculate iou of all bboxes with their associated anchors on all level

        Parameters
        ----------
        bboxes_coor: tf.Tensor, shape(n, 4) with 'n' is the number of bbox in an image

        Return
        ------
        List iou of all bboxes with their anchors on 4 levels

        '''
        iou_of_all_levels = []
        anchor_indexes_all_levels = []
        for level in range(4):
            find_associated_anchors_result = self.find_associated_anchors(bboxes_coor, level)
            anchor_indexes = find_associated_anchors_result[1] # (n, 2)
            anchors_coor = find_associated_anchors_result[0] # (n, 3, 4)

            bboxes_coor = tf.stack([bboxes_coor, bboxes_coor, bboxes_coor], axis=1) # shape(n, 3, 4)
            bboxes_area, anchors_area, inter_area = self.calculate_areas(bboxes_coor, anchors_coor)
            bboxes_coor = tf.unstack(bboxes_coor, axis=1)[0] # reshape bboxes from (n, 3, 4) -> (3, 4) for next iter
            union_area = bboxes_area + anchors_area - inter_area # (n, 3)
            iou = inter_area / union_area # (n, 3)
            iou_of_all_levels.append(iou)
            anchor_indexes_all_levels.append(anchor_indexes)
        
        iou_of_all_levels = tf.stack(iou_of_all_levels, axis=-2) # (n, 4, 3)
        anchor_indexes_all_levels = tf.stack(anchor_indexes_all_levels, axis=-2) # (n, 4, 2)
        return [iou_of_all_levels, anchor_indexes_all_levels] 
    
    def get_target(self, bboxes_tensor_with_padding):
        # Convert bboxes with padding to bboxes list
        num_of_bbox = bboxes_tensor_with_padding[0][0]
        num_of_bbox = tf.cast(num_of_bbox, dtype=tf.int32)
        bboxes_coor = bboxes_tensor_with_padding[1 : num_of_bbox + 1, :]

        classification_target = tf.zeros((160, 160, 3 * self.level)) 
        regression_target = tf.zeros((160, 160, 3 * self.level, 4))
        iouaware_target = tf.zeros((160, 160, 3 * self.level))
        
        calculate_iou_result = self.calculate_iou(bboxes_coor)
        anchor_indexes_all_levels = calculate_iou_result[1] # (n, 4, 2)
                            
        iou_of_all_levels = calculate_iou_result[0] # (n, 4, 3)
        iou_of_all_levels = tf.reshape(iou_of_all_levels, (-1, 3 * self.level)) # (n, 12)

        best_iou_each_box = tf.reduce_max(iou_of_all_levels, axis=-1) # (n, )

        filter_condition = best_iou_each_box > 0.35 # (n - x,) x is number of invalid bbox
        bboxes_coor = tf.boolean_mask(bboxes_coor, filter_condition)

        anchor_indexes_all_levels = tf.boolean_mask(anchor_indexes_all_levels, filter_condition) # (n - x, 4, 2) 
        iou_of_all_levels = tf.boolean_mask(iou_of_all_levels, filter_condition) # (n - x, 12)
        best_iou_each_box = tf.boolean_mask(best_iou_each_box, filter_condition) # (n - x)
        

        layer_of_best_iou_each_box = tf.argmax(iou_of_all_levels, axis=-1, output_type=tf.int32) # (n - x, ) from 0 to 11
        level_of_best_iou = tf.math.floordiv(layer_of_best_iou_each_box, 3) # (n - x, ) 0 corresponds to level 2
        gather_index = tf.stack([tf.range(tf.shape(level_of_best_iou)[0]), tf.cast(level_of_best_iou, tf.int32)], axis=-1) # (n - x, 2)
        anchor_index_of_best_iou = tf.gather_nd(anchor_indexes_all_levels, gather_index) # (n - x, 2)
        scatter_index = tf.concat([anchor_index_of_best_iou, tf.reshape(layer_of_best_iou_each_box, (-1, 1))], axis=-1) # (n - x, 3)
        

        classification_target = tf.tensor_scatter_nd_update(classification_target, scatter_index, tf.ones((bboxes_coor.shape[0],)))
        regression_target = tf.tensor_scatter_nd_update(regression_target, scatter_index, bboxes_coor)
        iouaware_target = tf.tensor_scatter_nd_update(iouaware_target, scatter_index, best_iou_each_box)
        

        target_level_2_dict = {'classification': classification_target[:, :, :3],
                               'regression': regression_target[:, :, :3],
                               'iouaware': iouaware_target[:, :, :3]}

        target_level_3_dict = {'classification': classification_target[:, :, 3:6][:80, :80],
                               'regression': regression_target[:, :, 3:6][:80, :80],
                               'iouaware': iouaware_target[:, :, 3:6][:80, :80]}

        target_level_4_dict = {'classification': classification_target[:, :, 6:9][:40, :40],
                               'regression': regression_target[:, :, 6:9][:40, :40],
                               'iouaware': iouaware_target[:, :, 6:9][:40, :40]}

        target_level_5_dict = {'classification': classification_target[:, :, 9:12][:20, :20],
                               'regression': regression_target[:, :, 9:12][:20, :20],
                               'iouaware': iouaware_target[:, :, 9:12][:20, :20]}

        return [target_level_2_dict, target_level_3_dict, target_level_4_dict, target_level_5_dict]
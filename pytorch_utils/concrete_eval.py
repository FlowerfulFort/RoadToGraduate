import os

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np

from .eval import Evaluator


class FullImageEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_batch(self, predicted, model, data, prefix=""):
        names = data['image_name']
        for i in range(len(names)):
            self.on_image_constructed(names[i], predicted[i,...], prefix)

    def save(self, name, prediction, prefix=""):
        cv2.imwrite(os.path.join(self.save_dir, prefix + name), (prediction * 255).astype(np.uint8))

class CropEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_mask = None
        self.current_prediction = None
        self.current_image_name = None

    def process_batch(self, predicted, model, data, intermediate_dir, image_name, prefix=""):
        names = data['image_name']
        config = self.config
        batch_geometry = self.parse_geometry(data['geometry'])
        for i in range(len(names)):
            name = names[i]
            geometry = batch_geometry[i]
            sx, sy = geometry['sx'], geometry['sy']
            pred = self.cut_border(np.squeeze(predicted[i,...]))
            if name != self.current_image_name:
                if self.current_image_name is None:
                    self.current_image_name = name
                else:
                    self.on_image_constructed(self.current_image_name, self.current_prediction / self.current_mask, intermediate_dir, image_name, prefix=prefix)
                self.construct_big_image(geometry)
            self.current_prediction[sy + self.border:sy + config.target_rows, sx + self.border:sx + config.target_cols] += pred
            self.current_mask[sy+self.border:sy + config.target_rows, sx + self.border:sx + config.target_cols] += 1
            self.current_image_name = name

    def parse_geometry(self, batch_geometry):
        rows = batch_geometry['rows'].numpy()
        cols = batch_geometry['cols'].numpy()
        sx = batch_geometry['sx'].numpy()
        sy = batch_geometry['sy'].numpy()
        geometries = []
        for idx in range(rows.shape[0]):
            geometry = {'rows': rows[idx],
                        'cols': cols[idx],
                        'sx': sx[idx],
                        'sy': sy[idx]}
            geometries.append(geometry)
        return geometries

    def construct_big_image(self, geometry):
        self.current_mask = np.zeros((geometry['rows'], geometry['cols']), np.uint8)
        self.current_prediction = np.zeros((geometry['rows'], geometry['cols']), np.float32)

    def save(self, name, prediction, intermediate_dir, image_name, prefix=""):
        index1 = name.rfind("_")
        name1 = name[:index1]
        index2 = name1.rfind("_")
        dir_name = os.path.join(intermediate_dir, image_name + '_mask')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        cv2.imwrite(os.path.join(dir_name, name), (prediction * 255).astype(np.uint8))
        
        #cv2.imwrite(os.path.join('/results', 'full', prefix + name), (prediction * 255).astype(np.uint8))
        
    def post_predict_action(self, intermediate_dir, image_name, prefix):
        self.on_image_constructed(self.current_image_name, self.current_prediction / self.current_mask, intermediate_dir, image_name, prefix=prefix)
        self.current_image_name = None

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import numpy as np
from common.skeleton import Skeleton

seed = 42
np.random.seed(seed)
random.seed(seed)


class MocapDataset:
    def __init__(self, fps, skeleton):
        self._skeleton = skeleton
        self._fps = fps
        self._data = None # Must be filled by subclass
        self._cameras = None # Must be filled by subclass
    
    def remove_joints(self, joints_to_remove):
        kept_joints = self._skeleton.remove_joints(joints_to_remove)
        for subject in self._data.keys():
            for action in self._data[subject].keys():
                s = self._data[subject][action]
                if 'positions' in s:
                    s['positions'] = s['positions'][:, kept_joints]
                
        
    def __getitem__(self, key):
        return self._data[key]
        
    def subjects(self):
        return self._data.keys()
    
    def fps(self):
        return self._fps
    
    def skeleton(self):
        return self._skeleton
        
    def cameras(self):
        return self._cameras
    
    def supports_semi_supervised(self):
        # This method can be overridden
        return False
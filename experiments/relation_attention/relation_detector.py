import torch
import torchvision.models as models
import torchvision

NUM_CLASSES = 2
CLASS_AGNOSTIC = True

class resnet101_rcnn_fpn_attention():
    def __init(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [256, 512, 1024, 2048]
    
    def rcnn(self, is_train=True):
        num_classes = NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        if is_train:
            
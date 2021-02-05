import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from torchvision.models.detection.roi_heads import RoIHeads, maskrcnn_loss, fastrcnn_loss, maskrcnn_inference, keypointrcnn_loss, keypointrcnn_inference
from torch.jit.annotations import Optional, List, Dict, Tuple

# from experiments.relation_attention.roi_heads import RoIHeads

TRAIN_RPN_POST_NMS_TOP_N = 2000
TEST_RPN_POST_NMS_TOP_N = 300
Nongt_dim = TEST_RPN_POST_NMS_TOP_N
Emb_dim = 64
# fc_dim 与 group相等，group默认为16
Fc_dim = 16
Feat_dim = 1024


def extract_position_matrix(proposals, nongt_dim):
    """ Extract position matrix

    Args:
        bbox: [num_boxes, 4]

    Returns:
        position_matrix: [num_boxes, nongt_dim, 4]
    """
    # 不需要index一列，只需要4个坐标值
    # list转tensor
    bbox = torch.tensor([item.detach().numpy() for item in proposals]).squeeze()
    # xmin、ymin、xmax、ymax的维度都是[num_boxes, 1]
    xmin = bbox[:, 0].unsqueeze(dim=1)
    ymin = bbox[:, 1].unsqueeze(dim=1)
    xmax = bbox[:, 2].unsqueeze(dim=1)
    ymax = bbox[:, 3].unsqueeze(dim=1)
    # 根据xmin、ymin、xmax、ymax计算得到中心点坐标center_x、center_y，宽bbox_width和高bbox_height
    # [num_fg_classes, num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [num_fg_classes, num_boxes, num_boxes]
    # delta_x的维度都是[num_boxes, num_boxes]，且该矩阵的对角线都是0
    delta_x = (center_x - center_x.t())/bbox_width
    delta_x = torch.log(torch.max(torch.abs(delta_x), torch.tensor(1e-3)))
    delta_y = (center_y - center_y.t()) / bbox_height
    delta_y = torch.log(torch.max(torch.abs(delta_y), torch.tensor(1e-3)))
    delta_width = torch.log(bbox_width/bbox_width.t())
    delta_height = torch.log(bbox_height/bbox_height.t())

    # concat_list是一个长度为4的列表，列表中的每个值的维度是[num_boxes, num_boxes]。
    concat_list = [delta_x, delta_y, delta_width, delta_height]

    for idx, sym in enumerate(concat_list):
        # [num_boxes, nongt_dim]
        sym = sym[:, :nongt_dim]
        # [num_boxes, nongt_dim, 1]
        concat_list[idx] = sym.unsqueeze(dim=2)

    # [num_boxes, nongt_dim, 4]
    position_matrix = torch.cat((concat_list[0], concat_list[1], concat_list[2], concat_list[3]), dim=2)

    return position_matrix

def extract_position_embedding(position_matrix, feat_dim, wave_length=1000):
    # position_matrix, [num_rois, nongt_dim, 4]
    # feat_range [0,1,2,3,4,5,6,7]
    feat_range = torch.arange(0, feat_dim/8)
    dim_mat = torch.pow(torch.full((1,), wave_length), (8. / feat_dim) * feat_range)
    # 1*1*1*8 [1., 2.37137365, 5.62341309, 13.33521461, 31.62277603, 74.98941803, 177.82794189, 421.69650269]
    dim_mat = dim_mat.view(1, 1, 1, -1)

    # position_matrix [num_rois, nongt_dim, 4, 1]
    position_matrix = (100.0*position_matrix).unsqueeze(dim=3)
    # div_mat [num_rois, nongt_dim, 4, 8]
    div_mat = position_matrix / dim_mat

    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # 在维度3对sin_mat和cos_mat做concat操作
    # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
    embedding = torch.cat((sin_mat, cos_mat), dim=3)
    # embedding, [num_rois, nongt_dim, feat_dim]
    embedding = embedding.view(embedding.shape[0], -1, feat_dim)
    return embedding

class relation_attention_module():
    def __init__(self, emb_dim, fc_dim, feat_dim, nongt_dim):
        # 用全连接层实现论文中公式5的max函数输入
        # 输入是position_embedding_reshape，得到维度为[num_rois * nongt_dim, fc_dim]
        self.pos_fc = nn.Linear(emb_dim, fc_dim)

        self.query = nn.Linear(feat_dim, 1024)
        self.key = nn.Linear(feat_dim, feat_dim)

        self.weighted_affinity = nn.Softmax(dim=2)

        self.linear_out = nn.Conv2d(fc_dim*feat_dim, 1024, 1, groups=fc_dim)
    def forward(self, roi_feat, position_embedding,
                                        nongt_dim=Nongt_dim, fc_dim=Fc_dim, feat_dim=Feat_dim,
                                        dim=(1024, 1024, 1024),
                                        group=16, index=1):
        """ Attetion module with vectorized version

                Args:
                    roi_feat: [num_rois, feat_dim]
                    position_embedding: [num_rois, nongt_dim, emb_dim]
                    nongt_dim:
                    fc_dim: should be same as group
                    feat_dim: dimension of roi_feat, should be same as dim[2]
                    dim: a 3-tuple of (query, key, output)
                    group:
                    index:

                Returns:
                    output: [num_rois, ovr_feat_dim, output_dim]
                """
        # 因为dim默认是(1024, 1024, 1024)，group默认是16，所以dim_group就是(64, 64, 64)。
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        # 在roi_feat的维度0上选取前nongt_dim的值，得到的nongt_roi_feat的维度是[nongt_dim, feat_dim]
        nongt_roi_feat = roi_feat[:nongt_dim, :]

        # 将[num_rois, nongt_dim, emb_dim]的position_embedding reshape
        emb_shape = position_embedding.shape
        # [num_rois * nongt_dim, emb_dim]
        position_embedding_reshape = position_embedding.view(emb_shape[0]*emb_shape[1], emb_shape[2])

        # position_feat_1, [num_rois * nongt_dim, fc_dim]
        position_feat_1 = F.relu(self.pos_fc(position_embedding_reshape))
        # aff_weight, [num_rois, nongt_dim, fc_dim]
        aff_weight = position_feat_1.view(-1, nongt_dim, fc_dim)
        # 几何权重, [num_rois, fc_dim, nongt_dim]
        aff_weight = aff_weight.permute(0, 2, 1)

        # multi head
        assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
        # 用全连接层得到q_data，全连接层参数对应论文中公式4的WQ，
        # roi_feat对应公式4的fA，维度[num_rois, feat_dim]。q_data:[num_rois, 1024]
        q_data = self.query(roi_feat)
        # [num_rois, group, dim_group[0]]，默认是[num_rois, 16, 64]，
        q_data_batch = q_data.view(-1, group, int(dim_group[0]))
        # [group, num_rois, dim_group[0]]，默认是[16, num_rois, 64]。
        q_data_batch = q_data_batch.permute(1, 0, 2)

        # 用全连接层得到k_data，全连接层参数对应论文中公式4的WK，
        # nongt_roi_feat对应公式4的fA，维度[nongt_dim, feat_dim]。k_data:[nongt_dim, 1024]
        k_data = self.key(nongt_roi_feat)
        # [nongt_dim, group, dim_group[1]]，默认是[nongt_dim, 16, 64]，
        k_data_batch = k_data.view(-1, group, int(dim_group[1]))
        # [group, nongt_dim, dim_group[1]]，默认是[16, nongt_dim, 64]。
        k_data_batch = k_data_batch.permute(1, 0, 2)

        v_data = nongt_roi_feat

        # 论文中公式4的矩阵乘法。
        # aff维度是[group, num_rois, nongt_dim]，默认是[16, num_rois, nongt_dim]。
        aff = torch.bmm(q_data_batch, k_data_batch.permute(0, 2, 1))
        # aff_scale, [group, num_rois, nongt_dim] 对应论文中公式4的除法
        aff_scale = (1.0 / torch.sqrt(torch.tensor(dim_group[1]))) * aff
        # [num_rois, group, nongt_dim]
        # 这个aff_scale就是论文中公式4的结果：wA
        aff_scale = aff_scale.permute(1, 0, 2)

        assert fc_dim == group, 'fc_dim != group'
        # weighted_aff, [num_rois, fc_dim, nongt_dim]
        # maximum对应论文中公式5,softmax实现公式3，而在softmax中
        # # 会对输入求指数（以e为底），而要达到论文中公式3的形式（e的指数只有wA，没有wG），
        # # 就要先对wGmn求log，这样再求指数时候就恢复成wG。简而言之就是e^(log(wG)+wA)=wG+e^(wA)。
        # # softmax实现论文中公式3的操作，axis设置为2表示在维度2上进行归一化。
        weighted_aff = torch.log(torch.max(aff_weight, torch.tensor(1e-6))) + aff_scale
        # [num_rois, fc_dim, nongt_dim]
        aff_softmax = self.weighted_affinity(weighted_aff)
        # [num_rois * fc_dim, nongt_dim]
        aff_softmax_reshape = aff_softmax.view(-1, nongt_dim)

        # 公式2
        # output_t, [num_rois * fc_dim, feat_dim] w和fA相乘
        output_t = torch.mm(aff_softmax_reshape, v_data)
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
        output_t = output_t.view(-1, fc_dim * feat_dim, 1, 1)

        # 公式2用dim[2]（默认是1024）的1*1卷积计算,卷积层的参数对应论文中公式2的WV
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.linear_out(output_t)
        # [num_rois, dim[2]]，
        # 加上groups的操作（group数量设置为fc_dim，默认是16，对应论文中的Nr参数）完成了concat所有的fR
        output = linear_out.squeeze()

        return output

class FasterRCNN(GeneralizedRCNN):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    Example::

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FasterRCNN
        >>> from torchvision.models.detection.rpn import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # FasterRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> # put the pieces together inside a FasterRCNN model
        >>> model = FasterRCNN(backbone,
        >>>                    num_classes=2,
        >>>                    rpn_anchor_generator=anchor_generator,
        >>>                    box_roi_pool=roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)

class TwoMLPHead(torch.nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

        # feat_dim默认是1024，对应前面全连接层的维度
        self.relation_attention_1 = relation_attention_module(emb_dim=Emb_dim, fc_dim=Fc_dim, nongt_dim=Nongt_dim,feat_dim=representation_size)
        self.relation_attention_2 = relation_attention_module(emb_dim=Emb_dim, fc_dim=Fc_dim, nongt_dim=Nongt_dim,feat_dim=representation_size)

    def forward(self, x, pos_emb):
        x = x.flatten(start_dim=1)

        x = self.fc6(x)
        attention_1 = self.relation_attention_1.forward(x, pos_emb)
        x = F.relu(x + attention_1)

        x = self.fc7(x)
        attention_2 = self.relation_attention_2.forward(x, pos_emb)
        x = F.relu(x + attention_2)

        return x

class RoIHeads(RoIHeads):
    def forward(self, features, proposals, image_shapes, targets=None):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # compute bbox embedding
        position_matrix = extract_position_matrix(proposals, Nongt_dim)
        position_embedding = extract_position_embedding(position_matrix, feat_dim=64)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features, position_embedding)
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses

class FRCNN_FPN_atten(FasterRCNN):

    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet101', False)
        super(FRCNN_FPN_atten, self).__init__(backbone, num_classes)



    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()

    # def predict_boxes(self, images, boxes):
    def forward(self, images, boxes):
        device = list(self.parameters())[0].device
        images = images.to(device)
        boxes = boxes.to(device)

        targets = None
        original_image_sizes = [img.shape[-2:] for img in images]

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        # proposals, proposal_losses = self.rpn(images, features, targets)
        from torchvision.models.detection.transform import resize_boxes
        boxes = resize_boxes(
            boxes, original_image_sizes[0], images.image_sizes[0])
        proposals = [boxes]

        position_matrix = extract_position_matrix(proposals, Nongt_dim)
        position_embedding = extract_position_embedding(position_matrix, feat_dim=64)

        box_features = self.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = self.box_head(box_features, position_embedding)
        class_logits, box_regression = self.roi_heads.box_predictor(
            box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # score_thresh = self.roi_heads.score_thresh
        # nms_thresh = self.roi_heads.nms_thresh

        # self.roi_heads.score_thresh = self.roi_heads.nms_thresh = 1.0
        # self.roi_heads.score_thresh = 0.0
        # self.roi_heads.nms_thresh = 1.0
        # detections, detector_losses = self.roi_heads(
        #     features, [boxes.squeeze(dim=0)], images.image_sizes, targets)

        # self.roi_heads.score_thresh = score_thresh
        # self.roi_heads.nms_thresh = nms_thresh

        # detections = self.transform.postprocess(
        #     detections, images.image_sizes, original_image_sizes)

        # detections = detections[0]
        # return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(
            pred_boxes, images.image_sizes[0], original_image_sizes[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        return pred_boxes, pred_scores

    def load_image(self, img):
        pass


if __name__ == '__main__':
    model = FRCNN_FPN_atten(num_classes=2)
    model.eval()
    img_path = '../../data\\1.jpg'

    print(os.listdir('./'))
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
    img = img.transpose(2,0,1)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    x = torch.randn((1, 3, 224, 224))
    y = model(img, [])

    tmp_dict = model.state_dict()

    print(y)
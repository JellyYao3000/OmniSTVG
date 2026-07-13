import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou


def _assignment_to_tensors(assignment):
    pred_idx, target_idx = assignment
    return (
        torch.as_tensor(pred_idx, dtype=torch.int64),
        torch.as_tensor(target_idx, dtype=torch.int64),
    )


def _solve_lsap(cost_matrix, sizes):
    cost_matrix = cost_matrix.cpu()
    assignments = [
        linear_sum_assignment(chunk[i])
        for i, chunk in enumerate(cost_matrix.split(sizes, -1))
    ]
    return [_assignment_to_tensors(assignment) for assignment in assignments]


def _bbox_cost(out_bbox, tgt_bbox, cost_bbox, cost_giou):
    l1_cost = torch.cdist(out_bbox, tgt_bbox, p=1)
    giou_cost = -generalized_box_iou(
        box_cxcywh_to_xyxy(out_bbox),
        box_cxcywh_to_xyxy(tgt_bbox),
    )
    return cost_bbox * l1_cost + cost_giou * giou_cost


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([target["labels"] for target in targets])
        tgt_bbox = torch.cat([target["boxes"] for target in targets])

        class_cost = -out_prob[:, tgt_ids]
        cost_matrix = (
            _bbox_cost(out_bbox, tgt_bbox, self.cost_bbox, self.cost_giou)
            + self.cost_class * class_cost
        )
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1)

        sizes = [len(target["boxes"]) for target in targets]
        return _solve_lsap(cost_matrix, sizes)


class HungarianMatcher2(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, out_bbox, tgt_bbox):
        batch_size, num_queries = out_bbox.shape[:2]

        out_bbox = out_bbox.flatten(0, 1)
        tgt_bbox = tgt_bbox.flatten(0, 1)
        cost_matrix = _bbox_cost(out_bbox, tgt_bbox, self.cost_bbox, self.cost_giou)
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1)

        return _solve_lsap(cost_matrix, [tgt_bbox.size(0)])


class HungarianMatcher4(nn.Module):
    def __init__(self, cost_class: float = 1):
        super().__init__()
        self.cost_class = cost_class

    @torch.no_grad()
    def forward(self, pred_logits, labels):
        batch_size, num_queries = pred_logits.shape[:2]

        out_prob = pred_logits.flatten(0, 1).softmax(-1)
        target_ids = torch.cat([labels])
        cost_matrix = self.cost_class * -out_prob[:, target_ids]
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1)

        return _solve_lsap(cost_matrix, labels.size(-1))

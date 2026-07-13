import numpy as np
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops.boxes import box_area
from models.net_utils import MLP, gen_sineembed_for_position
import math
from .position_encoding import SeqEmbeddingLearned, SeqEmbeddingSine
from .attention import MultiheadAttention
from ..bert_model.bert_module import BertLayerNorm, BertLayer_Cross
from easydict import EasyDict as EDict


def greater_than_indices(tensor, n):
    indices = torch.nonzero(tensor > n, as_tuple=False)
    return indices


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

class QueryDecoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.MODEL.CG.HIDDEN
        nhead = cfg.MODEL.CG.HEADS
        num_layers = cfg.MODEL.CG.DEC_LAYERS
        self.box_num = cfg.MODEL.BOX_NUM

        self.d_model = d_model
        self.query_pos_dim = cfg.MODEL.CG.QUERY_DIM
        self.nhead = nhead
        self.video_max_len = cfg.INPUT.MAX_VIDEO_LEN
        self.return_weights = cfg.SOLVER.USE_ATTN
        return_intermediate_dec = True

        self.template_generator = TemplateGenerator(cfg)

        self.decoder = PosDecoder(
            cfg,
            num_layers,
            return_intermediate=return_intermediate_dec,
            return_weights=self.return_weights,
            d_model=d_model,
            query_dim=self.query_pos_dim
        )

        self.time_decoder = TimeDecoder(
            cfg,
            num_layers,
            return_intermediate=return_intermediate_dec,
            return_weights=True,
            d_model=d_model
        )

        # The position embedding of global tokens
        if cfg.MODEL.CG.USE_LEARN_TIME_EMBED:
            self.time_embed = SeqEmbeddingLearned(self.video_max_len + 1, d_model)
        else:
            self.time_embed = SeqEmbeddingSine(self.video_max_len + 1, d_model)

        self.pos_fc = nn.Sequential(
            BertLayerNorm(256, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.ReLU(True),
            BertLayerNorm(4, eps=1e-12),
        )

        self.time_fc = nn.Sequential(
            BertLayerNorm(256, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(True),
            BertLayerNorm(256, eps=1e-12),
        )
        self.time_embed2 = None
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, encoded_info, vis_pos=None, query=None, targets=None):
        encoded_feature = encoded_info["encoded_feature"]  # len, n_frame, d_model
        encoded_mask = encoded_info["encoded_mask"]  # n_frame, len
        n_vis_tokens = encoded_info["fea_map_size"][0] * encoded_info["fea_map_size"][1]
        encoded_pos = vis_pos.flatten(2).permute(2, 0, 1)
        encoded_pos = torch.cat([encoded_pos, torch.zeros_like(encoded_feature[n_vis_tokens:])], dim=0)
        # the contextual feature to generate dynamic learnable anchors
        frames_cls = encoded_info["frames_cls"]  # [n_frames, d_model]
        videos_cls = encoded_info["videos_cls"]  # the video-level gloabl contextual token, b x d_model

        b = len(encoded_info["durations"])
        t = max(encoded_info["durations"])
        device = encoded_feature.device

        # pos_query, content_query = self.template_generator(frames_cls, videos_cls)  # [n_frame, 4]  [n_frame, 256]
        pos_query, content_query = self.pos_fc(frames_cls), self.time_fc(videos_cls)

        pos_query = pos_query.sigmoid().unsqueeze(1)  # [n_frames, bs, 4]
        content_query = content_query.expand(t, content_query.size(-1)).unsqueeze(1)  # [n_frames, bs, d_model]

        query_mask = torch.zeros(b, t).bool().to(device)
        query_time_embed = self.time_embed(t).repeat(1, b, 1)  # [n_frames, bs, d_model]

        tgt = query[1].reshape(1,1,256).repeat(t, 1, 1)

        outputs_time = self.time_decoder(
            query_tgt=tgt,
            query_content=content_query,  # n_queriesx(b*t)xF
            query_time=query_time_embed,
            query_mask=query_mask,
            encoded_feature=encoded_feature,
            encoded_pos=encoded_pos,  # n_tokensx(b*t)xF
            encoded_mask=encoded_mask
        )

        tgt2 = query[0].reshape(1,1,256).repeat(t* self.box_num, 1, 1)

        outputs_pos = self.decoder(
            query_tgt=tgt2,  # t x b x c
            pred_boxes=pos_query.repeat(1,self.box_num,1).reshape(-1, 1, 4),  # n_queriesx(b*t)xF
            query_time=query_time_embed.repeat(1,self.box_num,1).reshape(-1, 1, 256),
            query_mask=query_mask.repeat(1, self.box_num),  # bx(t*n_queries)
            encoded_feature=encoded_feature,  # n_tokens x n_frames x c
            encoded_pos=encoded_pos,  # n_tokens x n_frames x c
            encoded_mask=encoded_mask,  # n_frames * n_tokens
        )

        return outputs_pos, outputs_time


class PosDecoder(nn.Module):
    def __init__(self, cfg, num_layers, return_intermediate=False, return_weights=False, d_model=256, query_dim=4):
        super().__init__()
        self.layers = nn.ModuleList([PosDecoderLayer(cfg) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.return_weights = False
        self.query_dim = query_dim
        self.d_model = d_model

        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.type_embed = None
    
        self.gf_mlp = MLP(d_model, d_model, d_model, 2)
        self.gf_mlp2 = MLP(d_model, d_model, d_model, 2)
        self.fuse_linear = nn.Linear(d_model*2, d_model)
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None
        self.norm2 = nn.LayerNorm(d_model)

        self.theta_t = cfg.MODEL.CG.TEMP_THETA
        self.theta_s_gt = cfg.MODEL.CG.SPAT_GT_THETA
        self.theta_s = cfg.MODEL.CG.SPAT_THETA

    def forward(
            self,
            query_tgt: Optional[Tensor] = None,
            pred_boxes: Optional[Tensor] = None, 
            query_time: Optional[Tensor] = None, 
            query_mask: Optional[Tensor] = None, 
            encoded_feature: Optional[Tensor] = None,
            encoded_pos: Optional[Tensor] = None,
            encoded_mask: Optional[Tensor] = None,
    ):
        intermediate = []
        intermediate_weights = []
        ref_anchors = []  # the query pos is like t x b x 4
        type_list = []

        for layer_id, layer in enumerate(self.layers):
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(pred_boxes)
            query_pos = self.ref_point_head(query_sine_embed)  # generated the position embedding

            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(query_tgt)

            # apply transformation
            query_sine_embed = query_sine_embed[..., :self.d_model] * pos_transformation

            query_tgt, temp_weights = layer(
                query_tgt=query_tgt, query_pos=query_pos,
                query_time_embed=query_time, query_sine_embed=query_sine_embed, query_mask=query_mask,
                encoded_feature=encoded_feature, encoded_pos=encoded_pos, encoded_mask=encoded_mask,
                is_first=(layer_id == 0))

            # iter update
            if self.bbox_embed is not None:
                tmp = self.bbox_embed(query_tgt)
                new_pred_boxes = tmp.sigmoid()
                type = self.type_embed(query_tgt)
                ref_anchors.append(new_pred_boxes)
                type_list.append(type)
                pred_boxes = new_pred_boxes.detach()
           
            if self.return_intermediate:
                intermediate.append(self.norm(query_tgt))
                if self.return_weights:
                    intermediate_weights.append(temp_weights)
            
            query_tgt = query_tgt.reshape(-1, 1, 256)
            pred_boxes = pred_boxes.reshape(-1, 1, 4)

        if self.norm is not None:
            query_tgt = self.norm(query_tgt)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query_tgt)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                outputs = [
                    torch.stack(ref_anchors).transpose(1, 2),
                    torch.stack(type_list).transpose(1, 2),
                ]
            else:
                outputs = [
                    torch.stack(intermediate).transpose(1, 2),
                    pred_boxes.unsqueeze(0).transpose(1, 2)
                ]

        if self.return_weights:
            return outputs, torch.stack(intermediate_weights)
        else:
            return outputs

class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word (+ video), position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    ==> video features and word embeddings are merged together by summing up.
    """

    def __init__(self, config, add_postion_embeddings=True):
        super(BertEmbeddings, self).__init__()
        """add_postion_embeddings: whether to add absolute positional embeddings"""
        self.add_postion_embeddings = add_postion_embeddings
        self.fuse_embeddings = nn.Sequential(  # 3072->768
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(n_filters=config.hidden_size, max_len=2500)

        self.token_type_embeddings = nn.Embedding(3, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visual_features, l_v, l_t, l_r):
        video_embeddings = self.fuse_embeddings(visual_features)
        token_type_ids = torch.IntTensor([0] * l_v + [1] * l_t + [2] * l_r).to(video_embeddings.device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings1 = video_embeddings + token_type_embeddings
        if self.add_postion_embeddings:
            embeddings1 = self.position_embeddings(embeddings1)
        embeddings1 = self.LayerNorm(embeddings1)
        embeddings1 = self.dropout(embeddings1)
        return embeddings1  # (N, L, D)

class PosDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Decoder Self-Attention
        d_model = cfg.MODEL.CG.HIDDEN
        nhead = cfg.MODEL.CG.HEADS
        dim_feedforward = cfg.MODEL.CG.FFN_DIM
        dropout = cfg.MODEL.CG.DROPOUT
        activation = "relu"
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_qtime_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_ktime_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_qtime_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        self.from_scratch_cross_attn = cfg.MODEL.CG.FROM_SCRATCH
        self.cross_attn_image = None
        self.cross_attn = None
        self.tgt_proj = None
        self.box_num = cfg.MODEL.BOX_NUM

        if self.from_scratch_cross_attn:
            self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        else:
            self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
            self,
            query_tgt: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
            query_time_embed=None,
            query_sine_embed=None,
            query_mask: Optional[Tensor] = None,
            encoded_feature: Optional[Tensor] = None,
            encoded_pos: Optional[Tensor] = None,
            encoded_mask: Optional[Tensor] = None,
            is_first=False,
    ):
        # Apply projections here
        # shape: num_queries x batch_size x 256
        # ========== Begin of Self-Attention =============
        q_content = self.sa_qcontent_proj(query_tgt)  # target is the input of the first decoder layer. zero by default.
        q_time = self.sa_qtime_proj(query_time_embed)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(query_tgt)
        k_time = self.sa_ktime_proj(query_time_embed)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(query_tgt)

        q = q_content + q_time + q_pos
        k = k_content + k_time + k_pos

        q, weights = self.self_attn(q.reshape(-1, self.box_num, 256).permute(1,0,2), k.reshape(-1, self.box_num, 256).permute(1,0,2), value=v.reshape(-1, self.box_num, 256).permute(1,0,2))
        q = q.permute(1,0,2).reshape(-1, 1, 256)
        # Temporal Self attention
        tgt2, weights = self.self_attn2(q, k, value=v)
        tgt = query_tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = tgt.reshape(-1, self.box_num, 256)
        # ========== End of Self-Attention =============

        # ========== Begin of Cross-Attention =============
        # Time Aligned Cross attention
        t, b, c = tgt.shape    # b is the video number
        n_tokens, bs, f = encoded_feature.shape   # bs is the total frames in a batch
        assert f == c   # all the token dim should be same

        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(encoded_feature)
        v = self.ca_v_proj(encoded_feature)

        k_pos = self.ca_kpos_proj(encoded_pos)

        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos.reshape(-1,self.box_num,256)
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(t, b, self.nhead, c // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(t, b, self.nhead, c // self.nhead)

        if self.from_scratch_cross_attn:
            q = torch.cat([q, query_sine_embed], dim=3).view(t, b, c * 2)
        else:
            q = (q + query_sine_embed).view(t, b, c)
            q = q + self.ca_qtime_proj(query_time_embed)

        k = k.view(n_tokens, bs, self.nhead, f//self.nhead)
        k_pos = k_pos.view(n_tokens, bs, self.nhead, f//self.nhead)

        if self.from_scratch_cross_attn:
            k = torch.cat([k, k_pos], dim=3).view(n_tokens, bs, f * 2)
        else:
            k = (k + k_pos).view(n_tokens, bs, f)

        # extract the actual video length query
        device = tgt.device
        q_cross = q.permute(1,0,2)

        if self.from_scratch_cross_attn:
            tgt2, _ = self.cross_attn(
                query=q_cross,
                key=k,
                value=v
            )
        else:
            tgt2, _ = self.cross_attn_image(
                query=q_cross,
                key=k,
                value=v
            )

        tgt2 = tgt2.view(b, t, f).transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, weights


class TemplateGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.MODEL.CG.HIDDEN
        self.pos_query_dim = cfg.MODEL.CG.QUERY_DIM
        self.content_proj = nn.Linear(self.d_model, self.d_model)
        self.gamma_proj = nn.Linear(self.d_model, self.d_model)
        self.beta_proj = nn.Linear(self.d_model, self.d_model)
        self.anchor_proj = nn.Linear(self.d_model, self.pos_query_dim)

    def forward(self, frames_cls=None, videos_cls=None):
        gamma_vec = torch.tanh(self.gamma_proj(videos_cls))
        beta_vec = torch.tanh(self.beta_proj(videos_cls))
        pos_query = self.anchor_proj(gamma_vec * frames_cls + beta_vec)  # [n_frame, 4]
        content_query = self.content_proj(videos_cls)  # [b, d_model]
        content_query = content_query.expand(content_query.size(0) * frames_cls.size(0), content_query.size(1))

        return pos_query, content_query  # [n_frame, 4]  [n_frame, d_model]


class TimeDecoder(nn.Module):
    def __init__(self, cfg, num_layers, return_intermediate=False, return_weights=False, d_model=256):
        super().__init__()
        self.layers = nn.ModuleList([TimeDecoderLayer(cfg) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.return_weights = return_weights

    def forward(
            self,
            query_tgt: Optional[Tensor] = None,
            query_content: Optional[Tensor] = None,
            query_time: Optional[Tensor] = None,
            query_mask: Optional[Tensor] = None,
            encoded_feature: Optional[Tensor] = None,
            encoded_pos: Optional[Tensor] = None,
            encoded_mask: Optional[Tensor] = None
    ):
        intermediate = []
        intermediate_weights = []

        for _, layer in enumerate(self.layers):
            query_tgt, weights = layer(
                query_tgt=query_tgt,
                query_content=query_content,
                query_time=query_time,
                query_mask=query_mask,
                encoded_feature=encoded_feature,
                encoded_pos=encoded_pos,
                encoded_mask=encoded_mask
            )
            if self.return_intermediate:
                intermediate.append(self.norm(query_tgt))
                if self.return_weights:
                    intermediate_weights.append(weights)

        if self.norm is not None:
            query_tgt = self.norm(query_tgt)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query_tgt)

        if self.return_intermediate:
            return torch.stack(intermediate).transpose(1, 2)

class TimeDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.MODEL.CG.HIDDEN
        nhead = cfg.MODEL.CG.HEADS
        dim_feedforward = cfg.MODEL.CG.FFN_DIM
        dropout = cfg.MODEL.CG.DROPOUT
        activation = "relu"

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
            self,
            query_tgt: Optional[Tensor] = None,
            query_content: Optional[Tensor] = None,
            query_time: Optional[Tensor] = None,
            query_mask: Optional[Tensor] = None,
            encoded_feature: Optional[Tensor] = None,
            encoded_pos: Optional[Tensor] = None,
            encoded_mask: Optional[Tensor] = None
    ):
        q = k = self.with_pos_embed(query_tgt, query_time)

        # Temporal Self attention
        query_tgt2, weights = self.self_attn(q, k, value=query_tgt, key_padding_mask=query_mask)
        query_tgt = self.norm1(query_tgt + self.dropout1(query_tgt2))

        query_tgt2, _ = self.cross_attn_image(
            query=query_tgt.permute(1, 0, 2),
            key=self.with_pos_embed(encoded_feature, encoded_pos),
            value=encoded_feature,
            key_padding_mask=encoded_mask,
        )

        query_tgt2 = query_tgt2.transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf
        query_tgt = self.norm3(query_tgt + self.dropout3(query_tgt2))

        # FFN
        query_tgt2 = self.linear2(self.dropout(self.activation(self.linear1(query_tgt))))
        query_tgt = query_tgt + self.dropout4(query_tgt2)
        query_tgt = self.norm4(query_tgt)
        return query_tgt, weights


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

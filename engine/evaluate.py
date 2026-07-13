import torch
import torch.nn
from typing import Dict

from utils.misc import to_device
from utils.comm import synchronize, is_main_process
from utils.matcher import HungarianMatcher2, HungarianMatcher4

from tqdm import tqdm
import numpy as np

matcher = HungarianMatcher2()
matcher2 = HungarianMatcher4()

def tubelet_matcher(bbox_dict, position_dict, box_num, target_objects):
    frame_ids = sorted([fid for fid in bbox_dict])
    for idx in range(0, len(frame_ids) - 1):
        left_fid = frame_ids[idx]
        right_fid = frame_ids[idx + 1]
        index = matcher(torch.Tensor(bbox_dict[left_fid]), torch.Tensor(bbox_dict[right_fid]))
        assert index[0][0].tolist() ==list(range(box_num)), "The order is incorrect."
        bbox_dict[left_fid] =  torch.Tensor(bbox_dict[left_fid])[:,index[0][0]].tolist()
        bbox_dict[right_fid] = torch.Tensor(bbox_dict[right_fid])[:,index[0][1]].tolist()
        position_dict[left_fid] =  torch.Tensor(position_dict[left_fid])[:,index[0][0]].tolist()
        position_dict[right_fid] = torch.Tensor(position_dict[right_fid])[:,index[0][1]].tolist()

    tubelet_positions = torch.Tensor([v for _,v in position_dict.items()]).sum(0)
    index2 = matcher2(tubelet_positions, torch.LongTensor(target_objects))
    tube_index = list(dict(sorted({int(k[1]):int(k[0]) for k in zip(index2[0][0], index2[0][1])}.items())).values())
  
    new_bbox_dict = {key: torch.Tensor(value)[:, tube_index].tolist() for key, value in bbox_dict.items()}
    return new_bbox_dict


@torch.no_grad()
def linear_interp(bbox_dict):
    frame_ids = sorted([fid for fid in bbox_dict])
    if len(frame_ids) < 2:
        return bbox_dict
    for idx in range(0, len(frame_ids) - 1):
        left_fid = frame_ids[idx]
        right_fid = frame_ids[idx + 1]
        if right_fid - left_fid > 1:
            interval = right_fid - left_fid
            r = np.array(bbox_dict[right_fid])
            l = np.array(bbox_dict[left_fid])
            delta = (r-l) / 5
            for step in range(1, interval):
                bbox_dict[left_fid + step] = (l + step * delta).tolist()
    frame_ids = sorted([fid for fid in bbox_dict])
    assert max(frame_ids) - min(frame_ids) + 1 == len(frame_ids) 
    return {fid : bbox_dict[fid] for fid in frame_ids}

@torch.no_grad()
def single_forward(cfg, model, videos, texts, targets, device, postprocessor):
    durations = videos.durations
    box_num = cfg.MODEL.BOX_NUM
    position_length = cfg.MODEL.POSITION_LENGTH

    targets[0]["durations"] = durations
    outputs = model(videos, texts, targets)
    
    b = len(durations)
    t = max(durations)
    batch_img_size = [list(target['ori_size']) for target in targets]
    orig_target_sizes = [img_size for img_size in batch_img_size for _ in range(t)]
    orig_target_sizes = torch.tensor(orig_target_sizes,device=device)
   
    frames_ids = [target['frame_ids'] for target in targets] 
    pred_boxs, pred_steds = postprocessor(outputs, orig_target_sizes, frames_ids, durations, box_num)
    pred_boxs = pred_boxs.view(-1, box_num, 4)
    pred_positions = outputs['pred_box_position'].reshape(-1, cfg.MODEL.BOX_NUM, position_length)
  
    vids = [target['item_id'] for target in targets]
    bbox_pred, position_pred, temp_pred = {}, {}, {}
    
    for i_b in range(b):
        frames_id = frames_ids[i_b]
        bbox_pred[vids[i_b]] = {}
        position_pred[vids[i_b]] = {}
        assert durations[i_b] == len(frames_id)
        for idx in range(durations[i_b]):
            bbox_pred[vids[i_b]][frames_id[idx]] = [pred_boxs[idx].detach().cpu().tolist()]
            position_pred[vids[i_b]][frames_id[idx]] = [pred_positions[idx].detach().cpu().tolist()]
          
    for i_b in range(b):
        temp_pred[vids[i_b]] = {
            "sted": pred_steds[i_b]
        }
            
    return bbox_pred, temp_pred, position_pred
    

@torch.no_grad()
def do_eval(cfg, mode, logger, model, postprocessor, data_loader, evaluator, device):
    """
    Video Spatial-Temporal Grounding Evaluation
    """
    model.eval()
    logger.info("Start evaluation on the {} split of {} dataset".format(mode, cfg.DATASET.NAME))
    box_num = cfg.MODEL.BOX_NUM
    for _, batch_dict in enumerate(tqdm(data_loader)):
        videos = batch_dict['videos'].to(device)
        texts = batch_dict['texts']
        targets = to_device(batch_dict["targets"], device) 
        target_objects = [d['id'] for d in batch_dict["targets"][0]['boxs']]
        
        for i in range(len(targets)):
            if 'qtype' not in targets[i]:
                targets[i]['qtype'] = 'none'
        
        videos1 = videos.subsample(2, start_idx=0)
        targets1 = [{'text':texts, 'item_id': target['item_id'], 'ori_size': target['ori_size'],
                     'qtype': target['qtype'], 'frame_ids': target['frame_ids'][0::2], 'actioness':target['actioness'][0::2], "eval":True} for target in targets]

        videos2 = videos.subsample(2, start_idx=1)
        targets2 = [{'text':texts, 'item_id': target['item_id'], 'ori_size': target['ori_size'],
                     'qtype': target['qtype'], 'frame_ids': target['frame_ids'][1::2], 'actioness':target['actioness'][1::2], "eval":True} for target in targets]

        bbox_pred1, temp_pred1, position_pred1 = single_forward(cfg, model, videos1, texts,
                                targets1, device, postprocessor)
        bbox_pred2, temp_pred2, position_pred2 = single_forward(cfg, model, videos2, texts,
                                targets2, device, postprocessor)
        
        bbox_pred, temp_pred = {}, {}
        for vid in bbox_pred1:
            bbox_pred1[vid].update(bbox_pred2[vid])
            position_pred1[vid].update(position_pred2[vid])
            bbox_pred1[vid] = tubelet_matcher(bbox_pred1[vid], position_pred1[vid], box_num, target_objects)
            bbox_pred[vid] = linear_interp(bbox_pred1[vid])
            temp_pred[vid] = {'sted' : [min(temp_pred1[vid]['sted'][0], temp_pred2[vid]['sted'][0]),
                              max(temp_pred1[vid]['sted'][1], temp_pred2[vid]['sted'][1])]}
            if 'qtype' in temp_pred1[vid]:
                temp_pred[vid]['qtype'] = temp_pred1[vid]['qtype']

        evaluator.update(bbox_pred)
        evaluator.video_update(temp_pred)
    
    synchronize()
    evaluator.synchronize_between_processes()
    if is_main_process():
        logger.info(f"Complete the inference on {mode} split of {cfg.DATASET.NAME}")
    
    res = evaluator.summarize()
    return res
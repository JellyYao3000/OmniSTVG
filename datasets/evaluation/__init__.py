from .bostvg_eval import BOSTVGEvaluator

def build_evaluator(cfg, logger, mode):
    if cfg.DATASET.NAME == 'BOSTVG':
        return BOSTVGEvaluator(
            logger,
            cfg.DATA_DIR,
            mode,
            iou_thresholds=[0.3, 0.5],
            save_pred=(mode=='test'),
            save_dir=cfg.OUTPUT_DIR,
        )
    else:
        raise NotImplementedError

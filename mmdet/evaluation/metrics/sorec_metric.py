from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import get_local_path
from mmengine.logging import MMLogger

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import METRICS
from ..functional import bbox_overlaps


@METRICS.register_module()
class SorecMetric(BaseMetric):
    default_prefix: Optional[str] = 'sorec'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: str = 'bbox',
                 topk=(1, 5, 10),
                 iou_thrs: Sequence[float] = np.arange(0.1, 1.0, 0.05),
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.topk = topk
        #IoU thresholds from 0.1 to 0.95 with a step of 0.05
        self.iou_thrs = tuple(iou_thrs)  # Ensure it's always a tuple

        with get_local_path(ann_file) as local_path:
            self.coco = COCO(local_path)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        dataset2score = {
            'sorec': {iou_thr: {k: 0.0 for k in self.topk} for iou_thr in self.iou_thrs}
        }
        dataset2count = {'sorec': 0.0}

        for result in results:
            img_id = result['img_id']

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            assert len(ann_ids) == 1
            img_info = self.coco.loadImgs(img_id)[0]
            target = self.coco.loadAnns(ann_ids[0])

            target_bbox = target[0]['bbox']
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            iou = bbox_overlaps(result['bboxes'],
                                np.array(converted_bbox).reshape(-1, 4))
            
            dataset_name = img_info['dataset_name']
            if dataset_name == 'refcoco':
                dataset_name = 'sorec'
            for iou_thr in self.iou_thrs:
                for k in self.topk:
                    if max(iou[:k]) >= iou_thr:
                        dataset2score[dataset_name][iou_thr][k] += 1.0
            dataset2count[dataset_name] += 1.0

        for dataset_name, iou_dict in dataset2score.items():
            for iou_thr, topk_score_dict in iou_dict.items():
                for k in self.topk:
                    try:
                        topk_score_dict[k] /= dataset2count[dataset_name]
                    except Exception as e:
                        print(e)

        out_results = {}
        for dataset_name, iou_dict in dataset2score.items():
            for iou_thr, topk_score_dict in iou_dict.items():
                precision_values = [topk_score_dict[k] for k in self.topk]
                logger.info(
                    f'Dataset: {dataset_name}, IoU Threshold: {iou_thr} - '
                    f'Precision @ {self.topk}: {precision_values}'
                )
                for k in self.topk:
                    key = f'{dataset_name}_iou_{iou_thr}_precision@{k}'
                    out_results[key] = topk_score_dict[k]

        mean_precision = 0.0
        for dataset_name, iou_dict in dataset2score.items():
            precision_values = [iou_dict[0.50][k] for k in self.topk]
            mean_precision += sum(precision_values)
        out_results['mean_precision'] = mean_precision / (len(self.topk)*len(dataset2score))

        return out_results
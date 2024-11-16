import copy
import warnings
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, )
from ..layers.transformer.adapter_grounding_dino_layers import (
    GroundingDinoTransformerEncoderWithAdapter)

from mmengine.dataset.base_dataset import Compose
from .grounding_dino import GroundingDINO

from mmdet.models.piza_modules.time_embedding import TimeEmbedding
from mmdet.models.piza_modules.box_sequence_embedding import BBoxSequenceInferenceModel


@MODELS.register_module()
class PizaAdapterGroundingDINO(GroundingDINO):
    def __init__(self,
                 trainable_keys: List[str] = [], 
                 freeze_keys: List[str] = [],
                 use_zoom_embed=False,
                 use_bbox_history_encoder=False,
                 use_bbox_history_encoder_embed=False,
                 bbox_history_encoder_path=None,
                 use_multistep_prediction=True,
                 adapter_type='adapter-plus',
                 test_pipeline_in_predict=None,
                 test_pipeline_mean=None,
                 test_pipeline_std=None,
                 emb_in_ch = 256,
                 emb_out_ch = 256,
                 *args, **kwargs) -> None:
        self.use_zoom_embed = use_zoom_embed
        self.use_bbox_history_encoder = use_bbox_history_encoder
        self.use_bbox_history_encoder_embed = use_bbox_history_encoder_embed
        self.bbox_history_encoder_path = bbox_history_encoder_path
        self.use_multistep_prediction = use_multistep_prediction
        self.adapter_type = adapter_type
        self.use_embed_in_adapter = (self.use_bbox_history_encoder_embed or self.use_zoom_embed)
        assert not (use_bbox_history_encoder_embed and use_zoom_embed)

        self.pipeline = Compose(test_pipeline_in_predict)
        self.mean = test_pipeline_mean if test_pipeline_mean else [0,0,0]
        self.mean = torch.tensor(self.mean).unsqueeze(1).unsqueeze(1)
        self.std = test_pipeline_std if test_pipeline_std else [1,1,1]
        self.std = torch.tensor(self.std).unsqueeze(1).unsqueeze(1) 
        self.emb_in_ch = emb_in_ch
        self.emb_out_ch = emb_out_ch

        super().__init__(*args, **kwargs)

        if self.use_zoom_embed:
            self.zoom_embedding = TimeEmbedding(emb_in_ch, emb_out_ch)
        if self.bbox_history_encoder_path:
            self.bbox_encoder = BBoxSequenceInferenceModel(weight_path=self.bbox_history_encoder_path)

        self.set_trainable_params(trainable_keys, freeze_keys)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)

        encoder_config = {
            "adapter_type": self.adapter_type,
            "use_embed_in_adapter": self.use_embed_in_adapter,
            "emb_in_ch": self.emb_in_ch,
            "emb_out_ch": self.emb_out_ch,
            **self.encoder
        }

        self.encoder = GroundingDinoTransformerEncoderWithAdapter(**encoder_config)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

    def set_trainable_params(self, trainable_keys: List[str], freeze_keys: List[str] = []) -> None:
        print("=== Turning off gradients... ===")
        trainable_names = []
        if self.bbox_history_encoder_path is not None:
            for name, param in self.bbox_encoder.named_parameters():
                freeze_keys.append(name)
        for name, param in self.named_parameters():
            freeze_flag = False
            for freeze_key in freeze_keys:
                if freeze_key in name:
                    print('[off]', name)
                    param.requires_grad_(False)
                    freeze_flag = True
                    break
            if freeze_flag:
                continue

            for key in trainable_keys:
                if key in name:
                    trainable_names.append([name, param.requires_grad])
                    break
            else:
                print('[off]', name)
                param.requires_grad_(False)
        print('---------------------------------')
        for name, requires_grad in trainable_names:
            print('[keep]', name, f'(requires_grad={requires_grad})')
        print("=================================")

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
        zoom_val_list=None,
        bbox_history=None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
    
        zoom_emb = None
        eos = None
        is_training = batch_data_samples[0].get('dataset_mode') == 'VG'

        # PIZA processing
        if is_training:
            bbox_history = [
                torch.tensor(item.bbox_history).to(img_feats[0].device)
                for item in batch_data_samples
            ]
        eos, zoom_value, bbox_history_encoder_embed = self.bbox_encoder(bbox_history)
        if (not is_training) and eos[0]:
            return None, eos
        if self.use_embed_in_adapter:
            if self.use_bbox_history_encoder_embed:
                zoom_emb = bbox_history_encoder_embed
            else:
                zoom_emb = self.zoom_embedding(zoom_value)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict,
            text_dict=text_dict,
            zoom_emb=zoom_emb,
        )

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples
        )
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        if (not is_training) and self.bbox_history_encoder_path:
            return head_inputs_dict, eos
        else:
            return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict, zoom_emb=None) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'],
            zoom_emb=zoom_emb)
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def inference_with_crop(self, batch_inputs, batch_data_samples,
                            t=None, rescale=None, crop_box=None, bbox_history=None):
        pipeline_inputs = []
        if t is not None:
            zoom_val_list = [t]
        else:
            zoom_val_list = None

        if self.use_bbox_history_encoder and bbox_history is None:
            bbox_history = [[0, 0, batch_inputs.shape[3], batch_inputs.shape[2]]]
            if t == 1:
                x1,y1,x2,y2 = crop_box[0]
                bbox_history.append([x1, y1, x2-x1, y2-y1])
            bbox_history = torch.tensor([bbox_history]).to(batch_inputs.device)

        if crop_box is not None :
            new_batch_inputs = []
            for i,data_samples in enumerate(batch_data_samples):
                x1,y1,x2,y2 = crop_box[i]
                batch_input = batch_inputs[i, :, int(y1):int(y2), int(x1):int(x2)]
                new_batch_inputs.append(batch_input)
                new_batch_inputs = torch.stack(new_batch_inputs)
        else:
            new_batch_inputs = batch_inputs

        for i,data_samples in enumerate(batch_data_samples):
            data = dict(
                img_path=data_samples.img_path,
                img_id=data_samples.img_id,
                height=new_batch_inputs[i].shape[0],
                width=new_batch_inputs[i].shape[1],
                dataset_mode="refcoco",
                text=data_samples.text,
                custom_entities=data_samples.custom_entities,
                tokens_positive=data_samples.get('tokens_positive', -1),
                instances=[dict(bbox=data_samples.gt_instances.bboxes[i].tolist(
                ), bbox_label=data_samples.gt_instances.labels[i].cpu(), ignore_flag=0)],
                sample_idx=i,
                img=new_batch_inputs[i].cpu().numpy().astype(np.uint8).transpose(1, 2, 0),
                img_shape=new_batch_inputs[i].shape,
                ori_shape=new_batch_inputs[i].shape,
                batch_input_shape=new_batch_inputs[i].shape,
            )
            pipeline_inputs.append(data)

        pipeline_outputs = []
        for pipeline_input in pipeline_inputs:
            pipeline_outputs.append(self.pipeline(pipeline_input))
    
        batch_inputs = ((pipeline_outputs[0]["inputs"] - torch.tensor(self.mean)) / self.std).unsqueeze(0).cuda()
        batch_data_samples = [pipeline_outputs[0]["data_samples"]]
        batch_data_samples[0].batch_input_shape = batch_data_samples[0].img_shape

        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

                if self.bbox_history_encoder_path is not None:
                    head_inputs_dict, eos = self.forward_transformer(
                        copy.deepcopy(visual_feats), text_dict, batch_data_samples,
                        zoom_val_list, bbox_history)
                    if eos.item() == True:
                        return None, None, None, eos
                else:
                    head_inputs_dict = self.forward_transformer(
                        copy.deepcopy(visual_feats), text_dict, batch_data_samples,
                        zoom_val_list, bbox_history)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)
        if self.bbox_history_encoder_path is not None:
             return results_list, entities, is_rec_tasks, eos
        else:
            return results_list, entities, is_rec_tasks,

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        if self.use_multistep_prediction:
            MAX_ZOOM = 5
            crop_box = None
            results_list = None
            entities = None
            is_rec_tasks = None
            # Bbox history
            h = batch_inputs.shape[2]
            w = batch_inputs.shape[3]
            bbox_history = [[[0, 0, w, h]]]
            for pred_step in range(MAX_ZOOM):
                bbox_history_tensor = [torch.Tensor(bbox_seq).to(batch_inputs.device) for bbox_seq in bbox_history]
                if pred_step == 0:
                    _results_list, _entities, _is_rec_tasks, eos = self.inference_with_crop(
                        batch_inputs, batch_data_samples, 0, rescale=rescale, crop_box=None, bbox_history=bbox_history_tensor)
                else:
                    _results_list, _entities, _is_rec_tasks, eos = self.inference_with_crop(
                        batch_inputs, batch_data_samples, 0, rescale=rescale, crop_box=[crop_box], bbox_history=bbox_history_tensor)
                if eos.item():
                    break
                # Resize the bboxes
                if pred_step != 0:
                    for res in _results_list[0].bboxes:
                        for i in range(4):
                            res[i]+= bbox_history[0][-1][i%2]
                
                results_list = copy.deepcopy(_results_list)
                entities = _entities
                is_rec_tasks = _is_rec_tasks
                crop_box = _results_list[0].bboxes[0]
                # Update bbox history
                x1, y1, x2, y2 = results_list[0].bboxes[0].tolist()
                bbox_history = [bbox_history[0] + [[int(x1), int(y1), int(x2-x1), int(y2-y1)]]]

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
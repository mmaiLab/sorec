import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import ModuleList, BaseModule
from torch import Tensor

from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from mmdet.utils import ConfigType, OptConfigType
from .deformable_detr_layers import (DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer)
from .utils import get_text_sine_pos_embed

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

from mmdet.models.adapters.adapter_plus import AdapterPlus


class GroundingDinoTransformerEncoderWithAdapter(DeformableDetrTransformerEncoder):
    def __init__(self,
                 text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 adapter_type=None,
                 use_embed_in_adapter=False,
                 emb_in_ch=256,
                 emb_out_ch=256,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.use_embed_in_adapter = use_embed_in_adapter
        self.adapter_type = adapter_type
        self.emb_in_ch = emb_in_ch
        self.emb_out_ch = emb_out_ch
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayerWithAdapter(
                adapter_type=self.adapter_type,
                use_embed_in_adapter=self.use_embed_in_adapter,
                emb_in_ch=self.emb_in_ch,
                emb_out_ch=self.emb_out_ch,
                **self.layer_cfg
            )
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayerWithAdapter(
                adapter_type=self.adapter_type,
                emb_in_ch=self.emb_in_ch,
                emb_out_ch=self.emb_out_ch,
                **self.text_layer_cfg
            )
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None,
                zoom_emb=None):
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                    zoom_emb=zoom_emb,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                zoom_emb=zoom_emb)
        return output, memory_text


class DeformableDetrTransformerEncoderLayerWithAdapter(DeformableDetrTransformerEncoderLayer):
    def __init__(self,
                 adapter_type=None,
                 use_embed_in_adapter=False,
                 emb_in_ch=256,
                 emb_out_ch=256,
                 adapter_bottle_neck_dim=256,
                 *args, **kwargs) -> None:
        self.adapter_type = adapter_type
        self.use_embed_in_adapter = use_embed_in_adapter
        self.emb_in_ch = emb_in_ch
        self.emb_out_ch = emb_out_ch
        self.adapter_bottle_neck_dim = adapter_bottle_neck_dim
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, norms, and encoder adapter."""
        super()._init_layers()  # Inherit the self-attn, ffn, and norms initialization
        if self.adapter_type == 'adapter-plus':
            self.adapter1 = AdapterPlus(
                hidden_size=self.embed_dims,
                embedding_size=self.adapter_bottle_neck_dim,
                use_embed=self.use_embed_in_adapter,
                emb_in_ch=self.emb_in_ch,
                emb_out_ch=self.emb_out_ch,
            )
            self.adapter2 = AdapterPlus(
                hidden_size=self.embed_dims,
                embedding_size=self.adapter_bottle_neck_dim,
                use_embed=self.use_embed_in_adapter,
                emb_in_ch=self.emb_in_ch,
                emb_out_ch=self.emb_out_ch,
            )
            self.adapter1.initialize()
            self.adapter2.initialize()
        else:
            raise NotImplementedError

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, zoom_emb,
                **kwargs) -> Tensor:
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.adapter1(query, zoom_emb)  # adapter
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.adapter2(query, zoom_emb)  # adapter
        query = self.norms[1](query)

        return query


class DetrTransformerEncoderLayerWithAdapter(BaseModule):
    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256, num_heads=8, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None,
                 adapter_type=None,
                 use_embed_in_adapter=False,
                 emb_in_ch=256,
                 emb_out_ch=256,
                 adapter_bottle_neck_dim=256,
                 ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.adapter_type = adapter_type
        self.use_embed_in_adapter = use_embed_in_adapter
        self.emb_in_ch = emb_in_ch
        self.emb_out_ch = emb_out_ch
        self.adapter_bottle_neck_dim = adapter_bottle_neck_dim
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)
        if self.adapter_type == "adapter-plus":
            self.adapter1 = AdapterPlus(
                hidden_size=self.embed_dims,
                embedding_size=self.adapter_bottle_neck_dim,
                use_embed=self.use_embed_in_adapter,
                emb_in_ch=self.emb_in_ch,
                emb_out_ch=self.emb_out_ch,
            )
            self.adapter2 = AdapterPlus(
                hidden_size=self.embed_dims,
                embedding_size=self.adapter_bottle_neck_dim,
                use_embed=self.use_embed_in_adapter,
                emb_in_ch=self.emb_in_ch,
                emb_out_ch=self.emb_out_ch,
            )
            self.adapter1.initialize()
            self.adapter2.initialize()
        else:
            raise NotImplementedError

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, zoom_emb, **kwargs) -> Tensor:
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.adapter1(query, zoom_emb)  # adapter
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.adapter2(query, zoom_emb)  # adapter
        query = self.norms[1](query)

        return query

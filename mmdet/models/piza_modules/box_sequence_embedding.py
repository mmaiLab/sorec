import re
import os

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from mmdet.models.piza_modules.positional_encoding_emb import LearnableFourierFeatures

class BoxFeatureExtraction(nn.Module):
    def __init__(self):
        super(BoxFeatureExtraction, self).__init__()

    def compute_normalized_size(self, batch):
        # Compute area for each bounding box in the batch
        areas = batch[:, :, 2] * batch[:, :, 3]  # Shape: [num_sequences, num_boxes]
        normalized_size = areas / areas[:, [0]]  # Normalize by the first box's area in each sequence
        return normalized_size

    def compute_relative_size(self, batch):
        # Compute area for each bounding box in the batch
        areas = batch[:, :, 2] * batch[:, :, 3]  # Shape: [num_sequences, num_boxes]
        # Divide each box's area by the previous box's area along the sequence axis
        relative_size = torch.ones_like(areas)
        relative_size[:, 1:] = areas[:, 1:] / (areas[:, :-1]+1e-8)
        return relative_size

    def compute_normalized_width(self, batch):
        # Use the width of the first box in each sequence as the normalization factor
        W = batch[:, [0], 2]  # Shape: [num_sequences, 1]
        normalized_width = batch[:, :, 2] / W
        return normalized_width

    def compute_normalized_height(self, batch):
        # Use the height of the first box in each sequence as the normalization factor
        H = batch[:, [0], 3]  # Shape: [num_sequences, 1]
        normalized_height = batch[:, :, 3] / H
        return normalized_height

    def compute_center_position_x(self, batch):
        # Compute the center x position for each bounding box and normalize by W
        W = batch[:, [0], 2]  # Width of the first box in each sequence
        center_x = (batch[:, :, 0] + batch[:, :, 0] + batch[:, :, 2]) / (2 * W)
        return center_x

    def compute_center_position_y(self, batch):
        # Compute the center y position for each bounding box and normalize by H
        H = batch[:, [0], 3]  # Height of the first box in each sequence
        center_y = (batch[:, :, 1] + batch[:, :, 1] + batch[:, :, 3]) / (2 * H)
        return center_y

    def get_feat_dim1(self):
        return 2

    def get_feat_dim2(self):
        return 4

    def forward(self, batch):
        # Compute all features
        normalized_size = self.compute_normalized_size(batch)
        relative_size = self.compute_relative_size(batch)
        normalized_width = self.compute_normalized_width(batch)
        normalized_height = self.compute_normalized_height(batch)
        center_position_x = self.compute_center_position_x(batch)
        center_position_y = self.compute_center_position_y(batch)

        # Concatenate all features along the last dimension to form the final feature vector
        features = torch.cat([
            normalized_size.unsqueeze(-1),
            relative_size.unsqueeze(-1),
            normalized_width.unsqueeze(-1),
            normalized_height.unsqueeze(-1),
            center_position_x.unsqueeze(-1),
            center_position_y.unsqueeze(-1)
        ], dim=-1)
        
        # Output shape will be [num_sequences, num_boxes, num_features]
        return features

class BoxSequenceEmbedding(nn.Module):
    def __init__(self, feature_dim, out_dim = 512, h_dim = 32, nhead = 4, num_layers=1, use_mid_mlp=False):
        super(BoxSequenceEmbedding, self).__init__()

        self.pos_dim1 = 1
        self.pos_dim2 = 2
        self.box_feat = BoxFeatureExtraction()
        self.fdim1 = self.box_feat.get_feat_dim1()
        self.fdim2 = self.box_feat.get_feat_dim2()
        self.lff1 = LearnableFourierFeatures(pos_dim=self.pos_dim1, f_dim=h_dim//self.pos_dim1//2, h_dim=h_dim, d_dim=feature_dim//2//(self.fdim1//self.pos_dim1))
        self.lff2 = LearnableFourierFeatures(pos_dim=self.pos_dim2, f_dim=h_dim//self.pos_dim2//2, h_dim=h_dim, d_dim=feature_dim//2//(self.fdim2//self.pos_dim2))
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc3 = nn.Linear(feature_dim, out_dim)
        self.use_mid_mlp = use_mid_mlp
        if self.use_mid_mlp:
            self.mid_mlp = nn.Linear(8, 8)

    def preprocess(self, bbox_history):
        x = pad_sequence(bbox_history, batch_first=True)
        mask = (x.sum(dim=-1) == 0)
        x = self.box_feat(x)
        if self.use_mid_mlp:
            x = self.mid_mlp(x)

        pos1 = x[:,:,:self.fdim1]
        pos1 = pos1.view(pos1.size(0), pos1.size(1), self.fdim1//self.pos_dim1, self.pos_dim1)

        pos2 = x[:,:,self.fdim1:]
        pos2 = pos2.view(pos2.size(0), pos2.size(1), self.fdim2//self.pos_dim2, self.pos_dim2)
        return pos1, pos2, mask
    
    def forward(self, bbox_history):
        x1, x2, mask = self.preprocess(bbox_history)
        x1 = self.lff1(x1)
        x2 = self.lff2(x2)
        x = torch.cat([x1,x2],dim=2)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.fc3(x)
        valid_counts = (~mask).sum(dim=1, keepdim=True)
        x = (x * (~mask).unsqueeze(-1)).sum(dim=1) / valid_counts
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

class BBoxSequenceModel(nn.Module):
    def __init__(self, feature_dim=None, out_dim=None, h_dim=None, weight_path=None):
        super(BBoxSequenceModel, self).__init__()
        if weight_path is not None:
            values = re.findall(r'([fobeh])(\d+)', os.path.basename(weight_path))
            values = {key: int(value) for key, value in values}
            if 'f' in values:
                feature_dim = int(values['f'])
            if 'o' in values:
                out_dim = int(values['o'])
            if 'h' in values:
                h_dim = int(values['h'])
                
        if feature_dim == None:
            feature_dim = 32
        if out_dim== None:
            out_dim = 16
        if h_dim == None:
            h_dim = 16

        self.box_encoder = BoxSequenceEmbedding(feature_dim=feature_dim, out_dim=out_dim)

        self.binary_classifier = nn.Sequential(
            nn.Linear(out_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

        self.continuous_regressor = nn.Sequential(
            nn.Linear(out_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

        if weight_path:
            self.load_weights(weight_path)

    def load_weights(self, weight_path):
        state_dict = torch.load(weight_path)
        self.load_state_dict(state_dict)

    def forward(self, bbox_subsequence):
        features = self.box_encoder(bbox_subsequence)
        binary_output = self.binary_classifier(features).squeeze(-1)
        continuous_output = self.continuous_regressor(features).squeeze(-1)
        return binary_output, continuous_output, features


class BBoxSequenceInferenceModel(BBoxSequenceModel):
    def post_process(self, eos, zoom_value, feat, eos_thres=0.95):
        eos = eos >= eos_thres
        zoom_value = torch.round(zoom_value / 0.05) * 0.05
        return eos, zoom_value, feat.detach()

    def forward(self, *args, **kwargs):
        return self.post_process(*super().forward(*args, **kwargs))

from mmdet.models.detectors.piza_adapter_grounding_dino_demo import DemoPizaAdapterGroundingDINO
from mmengine.dataset.base_dataset import Compose
from mmengine.registry import init_default_scope
import argparse
import torch
from PIL import Image, ImageDraw
import numpy as np
from mmdet.structures import DetDataSample
from mmengine.config import Config, ConfigDict

def get_args():
    parser = argparse.ArgumentParser(description="Parse image path and reference text.")
    parser.add_argument("-i", "--img", type=str, help="Path to the input image.")
    parser.add_argument("-r", "--ref", type=str, required=True, help="Reference text.")
    return parser.parse_args()

def load_pretrained_model(model_path1='piza_adapter.pth',
                          model_path2='grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'):
    state_dict1 = torch.load(model_path1)
    state_dict2 = torch.load(model_path2)
    state_dict2['state_dict'].update(state_dict1['state_dict'])
    return state_dict2

def main(args):
    print("Starting inference...")
    init_default_scope("mmdet")
    cfg = Config.fromfile("configs/piza_adapter_grounding_dino/piza_adapter_grounding_dino_swin-t_finetune_5e_sorec_demo.py")
    # Model
    del cfg._cfg_dict.model.type
    model = DemoPizaAdapterGroundingDINO(**cfg._cfg_dict.model)
    state_dict = load_pretrained_model()["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    # Data preparation
    test_pipeline_custom = [
        dict(
            type='LoadImageFromFile', backend_args=None,
            imdecode_backend='pillow'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'text', 'custom_entities',
                    'tokens_positive'))
    ]
    pipeline_preprocess = Compose(test_pipeline_custom)
    image_org = Image.open(args.img)
    image = np.array(image_org)
    ref = args.ref
    pipeline_input = dict(
        img_path=args.img,
        dataset_mode="refcoco",
        text=ref,
        custom_entities=False,
        tokens_positive=None,
        sample_idx=0,
        img=image,
    )
    data = pipeline_preprocess(pipeline_input)
    batch_data_samples = [data["data_samples"]]
    batch_inputs = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cuda")
    with torch.no_grad():
        bbox_history = model.predict(batch_inputs, batch_data_samples)
    # Results 
    draw = ImageDraw.Draw(image_org)
    colors = ['blue', 'red', 'orange', 'magenta', 'cyan'] 
    bbox_history = bbox_history[0][1:]
    for i, bbox in enumerate(bbox_history):
        x1, y1, w, h = bbox
        draw.rectangle([x1, y1, x1+w, y1+h], outline=colors[i%len(colors)], width=4)
    print("bbox_history", bbox_history)
    image_org.save("sample_inference.jpg")
    print(f"Inference result saved successfully at: sample_inference.jpg !")

if __name__ == "__main__":
    args = get_args()
    main(args)

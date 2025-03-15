import numpy as np
from pprint import pprint
import yaml
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse
from utils.utils import save_model, Struct, set_seed, Wandb_Writer
import h5py
import time
from architecture.transformer import AttnMIL6 as AttnMIL
from architecture.clam import CLAM_SB, CLAM_MB
from architecture.transMIL import TransMIL
import torch
from wsi_core.WholeSlideImage import WholeSlideImage
import sys

def get_arguments():
    parser = argparse.ArgumentParser('Heatmap visualization', add_help=False)
    parser.add_argument('--config', dest='config', default='./config/camelyon_medical_ssl_config.yml',
                        help='settings of dataset in yaml format')
    parser.add_argument('--data_slide_dir', type=str, default='./source/')
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument(
        "--arch", type=str, default='ga', choices=['transmil', 'clam_sb', 'clam_mb',
                                                 'ga'], help="architecture"
    )


    # It is advisable to perform a preliminary check before executing the code.
    parser.add_argument(
        "--seed", type=int, default=3, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--device', default=0, type=int, help="CUDA device")
    parser.add_argument(
        "--n_masked_patch", type=int, default=0, help="whether use adversarial mask"
    )
    parser.add_argument(
        "--n_token", type=int, default=1, help="number of query token"
    )
    parser.add_argument(
        "--mask_drop", type=float, default=0, help="number of query token"
    )
    parser.add_argument("--zoom_factor", type=float, default=1.0,
                        help="determine the magnitude of zoom during visualizing the heatmap, range between 0 and 1")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = get_arguments()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    with open(args.config, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        config.update(vars(args))
        print("Used config:")
        pprint(config)
        conf = Struct(**config)

    group_name = f'ds_{conf.dataset}_{conf.pretrain}_arch_{conf.arch}_ntoken_{conf.n_token}_nmp_{conf.n_masked_patch}'
    ckpt_pth = './checkpoint-best.pth'
    vis_dir = os.path.join('./vis', group_name)
    os.makedirs(vis_dir, exist_ok=True)

    if conf.arch == 'transmil':
        net = TransMIL(conf)
    elif conf.arch == 'ga':
        net = AttnMIL(conf)
    elif conf.arch == 'clam_sb':
        net = CLAM_SB(conf, dropout=True)
    elif conf.arch == 'clam_mb':
        net = CLAM_MB(conf, dropout=True)
    else:
        print(f"Architecture {conf.arch} does not exist.")
        sys.exit(1)

    checkpoint = torch.load(ckpt_pth)
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    net.eval()

    h5_data = h5py.File(os.path.join(conf.data_dir, 'patch_feats_pretrain_%s.h5' % conf.pretrain), 'r')
    slide_names = list(h5_data.keys())
    for slide_id in slide_names:
        slide_data = h5_data[slide_id]
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)

        wsi_object = WholeSlideImage(slide_file_path)
        try:
            wsi_object.initXML(os.path.splitext(slide_file_path)[0] + '.xml')
        except FileNotFoundError:
            print(f'No XML annotations found for {slide_file_path}')

        feat = torch.from_numpy(slide_data['feat'][:]).unsqueeze(dim=0).to(device, dtype=torch.float32)
        coords = slide_data['coords'][:]

        _, _, attn_scores = net(feat, use_attention_mask=False)
        output_path = os.path.join(vis_dir, slide_id + '.png')
        probs = torch.softmax(attn_scores, dim=-1)[0].mean(0).cpu().numpy()
        probs = probs * probs.size * conf.zoom_factor
        heatmap = wsi_object.visHeatmap(scores=probs * 100, coords=coords, patch_size=(512, 512), segment=False, cmap='jet')
        heatmap.save(output_path)



if __name__ == '__main__':
    main()






import torch
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
from utils.utils import collate_features
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from models import build_model
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils import MetricLogger
import h5py
import openslide
import yaml
from utils.utils import Struct

device = 'cuda:{}'.format(0)

parser = argparse.ArgumentParser(description='Extract Features of Patches with TopK confidence')
parser.add_argument('--data_h5_dir', type=str, default='./patches')
parser.add_argument('--data_slide_dir', type=str, default='./source')
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default='./process_list_autogen.csv')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--config', dest='config', default='./config/bracs_medical_ssl_config.yml',
                    help='settings of Tip-Adapter in yaml format')
parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device to use')
args = parser.parse_args()
device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

def extract_feature(file_path, output_path, wsi, model, batch_size=8, verbose=1, print_every=20, pretrained=True, custom_downsample=1, target_patch_size=-1):
    
    print(f"Backbone Model: {cfg.backbone}")
    print(f"Extracting features from {file_path}")
    
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=16, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    feature_list = []
    coord_list = []
    for count, (batch, coords) in enumerate(loader):
        print(f"Processing batch {count + 1}/{len(loader)}")  
        with torch.no_grad():
            batch = batch.to(device, dtype=torch.float32)
            _, feature = model(batch, return_feature=True)
            feature_list.append(feature.cpu())
            coord_list.append(coords)

            
            if verbose > 0:
                print(f"Batch {count + 1}/{len(loader)}: feature_list 길이 = {len(feature_list)}, coord_list 길이 = {len(coord_list)}")
    features = torch.cat(feature_list, dim=0)
    coords = np.concatenate(coord_list, axis=0)

    
    if verbose > 0:
        print(f"최종 feature_list 길이: {len(feature_list)}")
        print(f"최종 coord_list 길이: {len(coord_list)}")
        if len(feature_list) > 0:
            print(f"feature_list 첫 번째 요소의 크기: {feature_list[0].size()}")
            print(f"coord_list 첫 번째 요소: {coord_list[0]}")

    return features.numpy(), coords

@torch.no_grad()
def extract_roi_features(model, cfg, output_dir):
    # dataloader
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_dataset = ImageFolder(os.path.join(cfg.data_dir, 'test'), transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.B,
                                            num_workers=cfg.n_worker,
                                            pin_memory=cfg.pin_memory,
                                            drop_last=False,
                                            shuffle=False
                                            )


    metric_logger = MetricLogger(delimiter="  ")
    header = 'Extract roi feature:'

    # switch to evaluation mode
    model.eval()
    feature_list = []
    label_list = []

    for batch in metric_logger.log_every(test_loader, 100, header):
        images = batch[0]
        target = batch[1]
        images = images.cuda()
        target = target.cuda()

        # compute output
        output, feature = model(images, return_feature=True)

        feature_list.append(feature.cpu())
        label_list.append(target.cpu())

    features = torch.cat(feature_list, dim=0)
    labels = torch.cat(label_list, dim=0)

    roi_feature_centroids = []
    for i in range(1, cfg.n_class):
        roi_feature = features[labels == i]
        roi_feature_centroids.append(roi_feature.mean(dim=0))
    roi_feature_centroids = torch.stack(roi_feature_centroids, dim=0)
    torch.save(roi_feature_centroids, os.path.join(output_dir, 'roi_feats.pt'))
    

if __name__ == '__main__':
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg = Struct(**cfg)
    print("\nRunning configs.")
    print(cfg, "\n")
    os.makedirs(cfg.data_dir, exist_ok=True)

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError
    bags_dataset = Dataset_All_Bags(csv_path)
    df = bags_dataset.df.set_index('slide_id')

    
    print('Printing some items from bags_dataset:')
    for i in range(5):  
        print(bags_dataset[i])

    print('loading model checkpoint')
    model = build_model(cfg)
    model = model.to(device)
    model.eval()
    total = len(bags_dataset)


    output_path = os.path.join(cfg.data_dir, 'patch_feats_pretrain_%s.h5'%cfg.pretrain)
    h5file = h5py.File(output_path, "w")
    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        if not os.path.exists(h5_file_path):
            continue
        slide_file_path = df.loc[slide_id]['full_path']
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        slide_feature, coords = extract_feature(h5_file_path, None, wsi,
                                            model=model, batch_size=args.batch_size, verbose=1, print_every=20,
                                            custom_downsample=args.custom_downsample,
                                            target_patch_size=args.target_patch_size)

        slide_grp = h5file.create_group(slide_id)
        slide_grp.create_dataset('feat', data=slide_feature.astype(np.float16))
        slide_grp.create_dataset('coords', data=coords)
        slide_grp.attrs['label'] = df.loc[slide_id]['label']
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(slide_id, time_elapsed))

    h5file.close()
    print("Stored features successfully!")
    print('Extracting ROI features and saving to roi_feats.pt')
    extract_roi_features(model, cfg, cfg.data_dir)
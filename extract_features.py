import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline,resnet18_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)
	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))
	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			mini_bs = coords.shape[0]
			features = model(batch)
			features = features.cpu().numpy()
			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--resnet_type', type=str, default="Resnet18")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--predict_model', type=str, default=None)
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError
	bags_dataset = Dataset_All_Bags(csv_path)	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
	print('loading model checkpoint')

	# resnet18
	# model = resnet50_baseline(pretrained=True)
	"""
	by lvyp(2023/2/14):
	修改的目的：加载之前用Resnet18 作为特征提取和分类器训练好的模型
	修改前："""
	#model = resnet18_baseline(pretrained=True)  # pretrained参数表示迁移学习
	"""修改后如下：
	
	model = resnet18_baseline(pretrained=False)
	net_dict = model.state_dict()
	predict_model = torch.load(args.predict_model)
	state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}  # 寻找网络中公共层，并保留预训练参数
	net_dict.update(state_dict)
	model.load_state_dict(net_dict)
	"""
	resnet_type = args.resnet_type
	if resnet_type == "Resnet50":
		print("运行Resnet50")
		model = resnet50_baseline(pretrained=True)
	elif resnet_type == "Resnet18":
		print("运行Resnet18")
		model = resnet18_baseline(pretrained=True)
	else:
		raise NotImplementedError
	model = model.to(device)

	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.eval()
	total = len(bags_dataset)
	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		image_name = slide_id.split("/")[-1]
		bag_name = image_name+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir,"patches", bag_name)
		slide_file_path = bags_dataset[bag_candidate_idx]
		slide_file_path = args.data_slide_dir + slide_file_path
		print(slide_file_path)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)
		if not args.no_auto_skip and image_name +'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue
		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		#print("各种路径：\n","路径1：h5_file_path",h5_file_path,"\n","路径2：wsi",wsi)
		output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
		model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")
		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))




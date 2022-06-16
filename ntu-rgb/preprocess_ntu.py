from torch.utils import data
import torch
import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# The npy files can be created following "https://github.com/shahroudy/NTURGB-D/tree/master/Python".

random.seed(1234)
np.random.seed(1234)

n_max = None

data_path1 = '/workspace/dataset/ntu-rgbd/skeletons/nturgbd_skeletons_s001_to_s017_npy'
data_path2 = '/workspace/dataset/ntu-rgbd/skeletons/nturgbd_skeletons_s018_to_s032_npy'

if n_max is not None:
	train_file = f'./dataset/ntu_rgb{n_max}/train.pkl'
	valid_file = f'./dataset/ntu_rgb{n_max}/valid.pkl'
	test_file = f'./dataset/ntu_rgb{n_max}/test.pkl'
else:
	train_file = f'./dataset/ntu_rgb/train.pkl'
	valid_file = f'./dataset/ntu_rgb/valid.pkl'
	test_file = f'./dataset/ntu_rgb/test.pkl'

os.makedirs(os.path.dirname(train_file), exist_ok=True)
os.makedirs(os.path.dirname(valid_file), exist_ok=True)
os.makedirs(os.path.dirname(test_file), exist_ok=True)

data_path1 = Path(data_path1)
data_path2 = Path(data_path2)

x_labels = []
y_labels = []
train = []
valid = []
test  = []
train_label = []
valid_label = []
test_label  = []

def ntu_to_florence(ntu_frames: np.ndarray):
	n_frame = len(ntu_frames)
	florance_frames = np.zeros((n_frame, 15, 3), dtype=np.float64)
	matching_list = [3,2,1,4,5, 6,8,9,10,12, 13,14,16,17,18]
	for fid, nid in enumerate(matching_list):
		florance_frames[:,fid,:] = ntu_frames[:,nid,:]
	return florance_frames

listdir1 =  os.listdir(data_path1) 
listdir2 =  os.listdir(data_path2) 

for i, npy_filename in tqdm(enumerate(listdir1 + listdir2)):
	if n_max is not None and i == n_max:
		break
	if not npy_filename.endswith(".npy"):
		print(f"[WARNING] Except file (not npy file): {npy_filename}")
		continue
	if i < len(listdir1):
		data = np.load(data_path1 / npy_filename, allow_pickle=True).item()
	else:
		data = np.load(data_path2 / npy_filename, allow_pickle=True).item()
		
	if not np.all(np.array(data['nbodys']) == 1):
		#print(f"[WARNING] Except file (nbody is not 1): {npy_filename}")
		continue
	frames = data['skel_body0']
	if len(frames) < 32:
		print(f"[WARNING] Except file (length of frames({len(frames)}) < 32): {npy_filename}")
		continue

	# TODO: Use all frame not only 32 frames
	indices = sorted(np.random.choice(len(frames), size=32, replace=False))
	frames = frames[indices]
	label = int(data['file_name'][-3:])
	if label > 120:
		print(f"[WARNING] Except file (label number ({label}) > 120): {npy_filename}")
		continue
	
	assert frames.shape == (32, 25, 3)
	frames = ntu_to_florence(frames)
	assert frames.shape == (32, 15, 3)
	frames = torch.from_numpy(frames)
	x_labels.append(frames)
	y_labels.append(label)

assert len(x_labels) == len(y_labels)
print(f"total number of data: {len(x_labels)}")
	
for x, y in zip(x_labels, y_labels):
	det = np.random.rand()
	if det < 0.8:
		train.append(x)
		train_label.append(y)
	elif det < 0.9:
		valid.append(x)
		valid_label.append(y)
	else:
		test.append(x)
		test_label.append(y)

print(f"total number of train data: {len(train)}")
print(f"total number of valid data: {len(valid)}")
print(f"total number of test  data: {len(test)}")

train_label = torch.from_numpy(np.asarray(train_label))
valid_label = torch.from_numpy(np.asarray(valid_label))
test_label  = torch.from_numpy(np.asarray(test_label))

# torch.save((torch.stack(train, 0), train_label), './dataset/ntu_rgb_100/train.pkl')
# torch.save((torch.stack(valid, 0), valid_label), './dataset/ntu_rgb_100/valid.pkl')
# torch.save((torch.stack(test, 0),  test_label),  './dataset/ntu_rgb_100/test.pkl')

torch.save((torch.stack(train, 0), train_label), train_file)
torch.save((torch.stack(valid, 0), valid_label), valid_file)
torch.save((torch.stack(test, 0),  test_label),  test_file)
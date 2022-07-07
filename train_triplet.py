import os
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from stgcn.model import *
from stgcn.metric import accuracy
from stgcn.config import get_args
from triplet_queue import TripletQueue

args = get_args()

os.makedirs(args.model_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

recons = args.recons
triplet = args.triplet

train_tensor, train_label = torch.load(args.train_path)
valid_tensor, valid_label = torch.load(args.valid_path)
test_tensor , test_label  = torch.load(args.test_path)
if args.use_2d:
	train_tensor = train_tensor[:,:,:,:2]
	valid_tensor = valid_tensor[:,:,:,:2]
	test_tensor = test_tensor[:,:,:,:2]
train_loader = data.DataLoader(data.TensorDataset(train_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=True, drop_last=True)
valid_loader = data.DataLoader(data.TensorDataset(valid_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=False, drop_last=True)
test_loader  = data.DataLoader(data.TensorDataset(test_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=False, drop_last=True)
train_label = train_label.to(device)
valid_label = valid_label.to(device)
test_label  = test_label.to(device)

A = [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
	 [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
	 [0,1,0,1,0,0,1,0,0,1,0,0,0,0,0],
	 [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
	 [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
	 [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
	 [0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
	 [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
	 [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
	 [0,0,1,0,0,0,0,0,0,0,1,0,1,0,0],
	 [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
	 [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
	 [0,0,0,0,0,0,0,0,0,1,0,0,1,0,0],
	 [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
	 [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]]
A = torch.from_numpy(np.asarray(A)).to(device)

if not recons:
	model = GGCN(A, train_tensor.size(3), args.num_classes, 
				[train_tensor.size(3), train_tensor.size(3)*3], [train_tensor.size(3)*3, 16, 32, 64], 
				args.feat_dims, args.dropout_rate, batch_size=args.batch_size)
else:
	model = AutoEncoderGGCN(A, train_tensor.size(3), args.num_classes, 
				[train_tensor.size(3), train_tensor.size(3)*3], [train_tensor.size(3)*3, 16, 32, 64], 
				args.feat_dims, args.dropout_rate, device, use_2d=args.use_2d, batch_size=args.batch_size)
if device == 'cuda':
	model.cuda()

num_params = 0
for p in model.parameters():
	num_params += p.numel()
print(model)
print('The number of parameters: {}'.format(num_params))

#criterion = nn.CrossEntropyLoss()
triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)
mse_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, 
					   betas=[args.beta1, args.beta2], weight_decay = args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma = 0.1)

best_epoch = 0
best_loss = 100

def train_triplet_recons_loss(model, x0, x1, x2):
	x0_recons, embedding0 = model.forward(x0)
	x1_recons, embedding1 = model.forward(x1)
	x2_recons, embedding2 = model.forward(x2)
	triplet_loss = triplet_criterion(embedding0, embedding1, embedding2)
	recons_loss = mse_criterion(x0, x0_recons.view(x0.shape))
	recons_loss += mse_criterion(x1, x1_recons.view(x1.shape))
	recons_loss += mse_criterion(x2, x2_recons.view(x2.shape))
	loss = triplet_loss + recons_loss
	return loss, triplet_loss, recons_loss

def train_triplet_loss(model, x0, x1, x2):
	embedding0 = model.encode(x0.float())
	embedding1 = model.encode(x1.float())
	embedding2 = model.encode(x2.float())
	triplet_loss = triplet_criterion(embedding0, embedding1, embedding2)
	return triplet_loss, triplet_loss

def train_recons_loss(model, x0, x1, x2):
	x0_recons, embedding0 = model.forward(x0)
	x1_recons, embedding1 = model.forward(x1)
	x2_recons, embedding2 = model.forward(x2)
	recons_loss = mse_criterion(x0, x0_recons.view(x0.shape))
	recons_loss += mse_criterion(x1, x1_recons.view(x1.shape))
	recons_loss += mse_criterion(x2, x2_recons.view(x2.shape))
	return recons_loss, recons_loss

def train():
	global best_epoch, best_loss

	if args.start_epoch:
		model.load_state_dict(torch.load(os.path.join(args.model_path, 
													  'model-%d.pkl'%(args.start_epoch))))

	# Training
	for epoch in range(args.start_epoch, args.num_epochs):
		losses = {'total': 0, 'triplet': 0, 'recons': 0}
		model.train()
		train_queue = TripletQueue()
		
		tqdm_train_loader = tqdm(train_loader)
		tqdm_train_loader.set_description(f'epoch{epoch+1}')
		for i, x in enumerate(tqdm_train_loader):
			target = train_label[i].to(torch.int64).cpu()
			train_queue.put(target, x[0])
			if not train_queue.hasnext():
				continue
			x0, x1, x2 = train_queue.next()
			#breakpoint()
			x0 = x0.float()
			x1 = x1.float()
			x2 = x2.float()
			
			if triplet and recons:
				loss, triplet_loss, recons_loss = train_triplet_recons_loss(model, x0, x1, x2)
				losses['triplet'] += triplet_loss.item()
				losses['recons'] += recons_loss.item()
			elif triplet:
				loss, triplet_loss = train_triplet_loss(model, x0, x1, x2)
				losses['triplet'] += triplet_loss.item()
			elif recons:
				loss, recons_loss = train_recons_loss(model, x0, x1, x2)
				losses['recons'] += recons_loss.item()
			else:
				raise NotImplementedError
			losses['total'] += loss.item()
			tqdm_train_loader.set_postfix({key: "{:.3f}".format(item/(i+1)) for key, item in losses.items()})
			model.zero_grad()
			loss.backward()
			optimizer.step()

		scheduler.step()
		# if triplet and recons:
		# 	print(f'[epoch{epoch+1}] loss: {total_loss/(i+1)} triplet: {total_triplet_loss/(i+1)} recons: {total_recons_loss/(i+1)}')
		# elif triplet:
		# 	print(f'[epoch{epoch+1}] loss: {total_loss/(i+1)} triplet: {total_triplet_loss/(i+1)}')
		# elif recons:
		# 	print(f'[epoch{epoch+1}] loss: {total_loss/(i+1)} recons: {total_recons_loss/(i+1)}')
		# else:
		# 	raise NotImplementedError

		if (epoch+1) % args.val_step == 0:
			model.eval()
			losses = {'total': 0, 'triplet': 0, 'recons': 0}

			valid_queue = TripletQueue()

			with torch.no_grad():
				tqdm_valid_loader = tqdm(valid_loader)
				tqdm_valid_loader.set_description(f'valid')
				for i, x in enumerate(tqdm_valid_loader):
					target = valid_label[i].to(torch.int64).cpu()
					valid_queue.put(target, x[0])
					if not valid_queue.hasnext():
						continue
					x0, x1, x2 = valid_queue.next()
					
					x0 = x0.float()
					x1 = x1.float()
					x2 = x2.float()

					if triplet and recons:
						loss, triplet_loss, recons_loss = train_triplet_recons_loss(model, x0, x1, x2)
						losses['triplet'] += triplet_loss.item()
						losses['recons'] += recons_loss.item()
					elif triplet:
						loss, triplet_loss = train_triplet_loss(model, x0, x1, x2)
						losses['triplet'] += triplet_loss.item()
					elif recons:
						loss, recons_loss = train_recons_loss(model, x0, x1, x2)
						losses['recons'] += recons_loss.item()
					else:
						raise NotImplementedError
					losses['total'] += loss.item()
					tqdm_valid_loader.set_postfix({key: "{:.3f}".format(item/(i+1)) for key, item in losses.items()})

				if best_loss > (losses['total']/(i+1)):
					best_epoch = epoch+1
					best_loss = (losses['total']/(i+1))
					torch.save(model.state_dict(), os.path.join(args.model_path, 'model-%d.pkl'%(best_epoch)))
					print(f"save model-{best_epoch}.pkl")

				# if triplet and recons:
				# 	print(f'[Valid]  loss: {total_loss/(i+1)} triplet: {total_triplet_loss/(i+1)} recons: {total_recons_loss/(i+1)}')
				# elif triplet:
				# 	print(f'[Valid]  loss: {total_loss/(i+1)} triplet: {total_triplet_loss/(i+1)}')
				# elif recons:
				# 	print(f'[Valid]  loss: {total_loss/(i+1)} recons: {total_recons_loss/(i+1)}')
				# else:
				# 	raise NotImplementedError

def test():
	global best_epoch

	# model = AutoEncoderGGCN(A, train_tensor.size(3), args.num_classes, 
	# 		 [train_tensor.size(3), train_tensor.size(3)*3], [train_tensor.size(3)*3, 16, 32, 64], 
	# 		 args.feat_dims, args.dropout_rate, device)
	model.load_state_dict(torch.load(os.path.join(args.model_path, 
												  'model-%d.pkl'%(best_epoch))))
	print("load model from 'model-%d.pkl'"%(best_epoch))

	model.eval()
	losses = {'total': 0, 'triplet': 0, 'recons': 0}
	
	test_queue = TripletQueue()
	with torch.no_grad():
		tqdm_test_loader = tqdm(test_loader)
		tqdm_test_loader.set_description(f'test')
		for i, x in enumerate(tqdm_test_loader):
			target = test_label[i].to(torch.int64).cpu()
			test_queue.put(target, x[0])
			if not test_queue.hasnext():
				continue
			x0, x1, x2 = test_queue.next()
			x0 = x0.float()
			x1 = x1.float()
			x2 = x2.float()
			
			if triplet and recons:
				loss, triplet_loss, recons_loss = train_triplet_recons_loss(model, x0, x1, x2)
				losses['triplet'] += triplet_loss.item()
				losses['recons'] += recons_loss.item()
			elif triplet:
				loss, triplet_loss = train_triplet_loss(model, x0, x1, x2)
				losses['triplet'] += triplet_loss.item()
			elif recons:
				loss, recons_loss = train_recons_loss(model, x0, x1, x2)
				losses['recons'] += recons_loss.item()
			else:
				raise NotImplementedError
			losses['total'] += loss.item()
			tqdm_test_loader.set_postfix({key: "{:.3f}".format(item/(i+1)) for key, item in losses.items()})
			
	# if triplet and recons:
	# 	print(f'[Test]   loss: {total_loss/(i+1)} triplet: {total_triplet_loss/(i+1)} recons: {total_recons_loss/(i+1)}')
	# elif triplet:
	# 	print(f'[Test]   loss: {total_loss/(i+1)} triplet: {total_triplet_loss/(i+1)}')
	# elif recons:
	# 	print(f'[Test]   loss: {total_loss/(i+1)} recons: {total_recons_loss/(i+1)}')
	# else:
	# 	raise NotImplementedError

if __name__ == '__main__':
	if args.mode == 'train':
		train()
	elif args.mode == 'test':
		best_epoch = args.test_epoch
	test()

# python main.py --data_path dataset\ntu_rgb --mode test --model_path models/ntu --test_epoch 68 --num_classes 120

# python main.py --data_path .\dataset\Florence_3d_actions --num_classes 9 --num_epoch 10
# 
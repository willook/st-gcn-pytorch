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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

triplet = False
train_tensor, train_label = torch.load(args.train_path)
valid_tensor, valid_label = torch.load(args.valid_path)
test_tensor , test_label  = torch.load(args.test_path)
train_loader = data.DataLoader(data.TensorDataset(train_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=False)
valid_loader = data.DataLoader(data.TensorDataset(valid_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=False)
test_loader  = data.DataLoader(data.TensorDataset(test_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=False)
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


recons = args.recons
if not recons:
	model = GGCN(A, train_tensor.size(3), args.num_classes, 
				[train_tensor.size(3), train_tensor.size(3)*3], [train_tensor.size(3)*3, 16, 32, 64], 
				args.feat_dims, args.dropout_rate)
else:
	model = AutoEncoderGGCN(A, train_tensor.size(3), args.num_classes, 
				[train_tensor.size(3), train_tensor.size(3)*3], [train_tensor.size(3)*3, 16, 32, 64], 
				args.feat_dims, args.dropout_rate, device)
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
def train():
	global best_epoch, best_loss

	if args.start_epoch:
		model.load_state_dict(torch.load(os.path.join(args.model_path, 
													  'model-%d.pkl'%(args.start_epoch))))

	# Training
	for epoch in range(args.start_epoch, args.num_epochs):
		train_loss = 0
		train_tloss = 0
		train_rloss = 0
		
		model.train()
		train_queue = TripletQueue()
		for i, x in tqdm(enumerate(train_loader)):
			target = train_label[i].to(torch.int64).cpu()
			train_queue.put(target, x[0])
			if not train_queue.hasnext():
				continue
			x0, x1, x2 = train_queue.next()
			x0 = x0.float()
			x1 = x1.float()
			x2 = x2.float()
			
			if recons:
				x0_recons, embedding0 = model.forward(x0)
				x1_recons, embedding1 = model.forward(x1)
				x2_recons, embedding2 = model.forward(x2)
				if triplet:
					triplet_loss = triplet_criterion(embedding0, embedding1, embedding2)
				recons_loss0 = mse_criterion(x0, x0_recons.view(x0.shape))
				recons_loss1 = mse_criterion(x1, x1_recons.view(x1.shape))
				recons_loss2 = mse_criterion(x2, x2_recons.view(x2.shape))
				recons_loss = recons_loss0 + recons_loss1 + recons_loss2
				if triplet:
					loss = triplet_loss + recons_loss
					train_tloss += triplet_loss.item()
				else:
					loss = recons_loss
				train_rloss += recons_loss.item()
			else:
				logit0 = model.encode(x0.float())
				logit1 = model.encode(x1.float())
				logit2 = model.encode(x2.float())
				loss = triplet_criterion(logit0, logit1, logit2)
			
			model.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			#train_acc  += accuracy(logit, target.view(1))
		
		scheduler.step()
		if recons:
			# print('[epoch',epoch+1,'] Train loss:',train_loss/(i+1), 'triplet:',train_tloss/(i+1), 'recons:',train_rloss/(i+1))
			print('[epoch',epoch+1,'] Train loss:',train_loss/(i+1), 'recons:',train_rloss/(i+1))
		else:
			print('[epoch',epoch+1,'] Train loss:',train_loss/(i+1))

		if (epoch+1) % args.val_step == 0:
			model.eval()
			val_loss = 0
			val_tloss = 0
			val_rloss = 0

			valid_queue = TripletQueue()
			with torch.no_grad():
				for i, x in tqdm(enumerate(valid_loader)):
					target = valid_label[i].to(torch.int64).cpu()
					valid_queue.put(target, x[0])
					if not valid_queue.hasnext():
						continue
					x0, x1, x2 = valid_queue.next()
					
					x0 = x0.float()
					x1 = x1.float()
					x2 = x2.float()
					if recons:
						x0_recons, embedding0 = model.forward(x0)
						x1_recons, embedding1 = model.forward(x1)
						x2_recons, embedding2 = model.forward(x2)
						
						if triplet:
							triplet_loss = triplet_criterion(embedding0, embedding1, embedding2)
						recons_loss0 = mse_criterion(x0, x0_recons.view(x0.shape))
						recons_loss1 = mse_criterion(x1, x1_recons.view(x1.shape))
						recons_loss2 = mse_criterion(x2, x2_recons.view(x2.shape))
						recons_loss = recons_loss0 + recons_loss1 + recons_loss2
						if triplet:
							loss = triplet_loss + recons_loss
							val_tloss += triplet_loss.item()
						else:
							loss = recons_loss
						val_rloss += recons_loss.item()
					else:
						logit0 = model.encode(x0.float())
						logit1 = model.encode(x1.float())
						logit2 = model.encode(x2.float())
						loss = triplet_criterion(logit0, logit1, logit2)
					val_loss += loss.item()
				if best_loss > (val_loss/i):
					best_epoch = epoch+1
					best_loss = (val_loss/i)
					torch.save(model.state_dict(), os.path.join(args.model_path, 'model-%d.pkl'%(best_epoch)))
					print(f"save model-{best_epoch}.pkl")

			if recons:
				# print('Val loss:',val_loss/(i+1), 'triplet:',val_tloss/(i+1), 'recons:',val_rloss/(i+1))
				print('Val loss:',val_loss/(i+1), 'recons:',val_rloss/(i+1))
			else:
				print('Val loss:',val_loss/(i+1))

def test():
	global best_epoch

	model = AutoEncoderGGCN(A, train_tensor.size(3), args.num_classes, 
			 [train_tensor.size(3), train_tensor.size(3)*3], [train_tensor.size(3)*3, 16, 32, 64], 
			 args.feat_dims, args.dropout_rate, device)
	model.load_state_dict(torch.load(os.path.join(args.model_path, 
												  'model-%d.pkl'%(best_epoch))))
	print("load model from 'model-%d.pkl'"%(best_epoch))

	model.eval()
	test_loss = 0
	
	test_queue = TripletQueue()
	with torch.no_grad():
		for i, x in tqdm(enumerate(test_loader)):
			target = test_label[i].to(torch.int64).cpu()
			test_queue.put(target, x[0])
			if not test_queue.hasnext():
				continue
			x0, x1, x2 = test_queue.next()
			x0 = x0.float()
			x1 = x1.float()
			x2 = x2.float()
			if recons:
				x0_recons, embedding0 = model.forward(x0)
				x1_recons, embedding1 = model.forward(x1)
				x2_recons, embedding2 = model.forward(x2)
				
				if triplet:
					triplet_loss = triplet_criterion(embedding0, embedding1, embedding2)
				recons_loss0 = mse_criterion(x0, x0_recons.view(x0.shape))
				recons_loss1 = mse_criterion(x1, x1_recons.view(x1.shape))
				recons_loss2 = mse_criterion(x2, x2_recons.view(x2.shape))
				if triplet:
					loss = triplet_loss + recons_loss0 + recons_loss1 + recons_loss2
				else:
					loss = recons_loss0 + recons_loss1 + recons_loss2
			else:
				logit0 = model.encode(x0.float())
				logit1 = model.encode(x1.float())
				logit2 = model.encode(x2.float())
				loss = triplet_criterion(logit0, logit1, logit2)
			
			test_loss += loss.item()
			
	print('Test loss:',test_loss/(i+1))

if __name__ == '__main__':
	if args.mode == 'train':
		train()
	elif args.mode == 'test':
		best_epoch = args.test_epoch
	test()

# python main.py --data_path dataset\ntu_rgb --mode test --model_path models/ntu --test_epoch 68 --num_classes 120

# python main.py --data_path .\dataset\Florence_3d_actions --num_classes 9 --num_epoch 10
# 
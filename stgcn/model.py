import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from stgcn.layer import GraphConvolution, StandConvolution

class GGCN(nn.Module):
	def __init__(self, adj, num_v, num_classes, gc_dims, sc_dims, feat_dims, dropout=0.5, batch_size=1):
		super(GGCN, self).__init__()
		torch.manual_seed(0)
		terminal_cnt = 5
		actor_cnt = 1
		adj = adj + torch.eye(adj.size(0)).to(adj).detach()
		ident = torch.eye(adj.size(0)).to(adj)
		zeros = torch.zeros(adj.size(0), adj.size(1)).to(adj)
		self.adj = torch.cat([torch.cat([adj, ident, zeros], 1),
							  torch.cat([ident, adj, ident], 1),
							  torch.cat([zeros, ident, adj], 1)], 0).float()
		self.terminal = nn.Parameter(torch.randn(terminal_cnt, actor_cnt, feat_dims))

		self.gcl = GraphConvolution(gc_dims[0]+feat_dims, gc_dims[1], num_v, dropout=dropout)
		self.conv= StandConvolution(sc_dims, num_classes, dropout=dropout)
		self.batch_size = batch_size
		nn.init.xavier_normal_(self.terminal)

	def forward(self, x):
		head_la = F.interpolate(torch.stack([self.terminal[0],self.terminal[1]],2), 6)
		head_ra = F.interpolate(torch.stack([self.terminal[0],self.terminal[2]],2), 6)
		lw_ra = F.interpolate(torch.stack([self.terminal[3],self.terminal[4]],2), 6)
		node_features = torch.cat([
								   (head_la[:,:,:3] + head_ra[:,:,:3])/2,
								   torch.stack((lw_ra[:,:,2], lw_ra[:,:,1], lw_ra[:,:,0]), 2),
								   lw_ra[:,:,3:], head_la[:,:,3:], head_ra[:,:,3:]], 2).to(x)
		x = torch.cat((x, node_features.permute(0,2,1).unsqueeze(1).repeat(1,32,1,1)), 3)

		concat_seq = torch.cat([x[:,:-2], x[:,1:-1], x[:,2:]], 2) # 1, 30, 45, 3
		multi_conv = self.gcl(self.adj, concat_seq)
		logit, _ = self.conv(multi_conv)
		
		return logit

	def encode(self, x):
		head_la = F.interpolate(torch.stack([self.terminal[0],self.terminal[1]],2), 6)
		head_ra = F.interpolate(torch.stack([self.terminal[0],self.terminal[2]],2), 6)
		lw_ra = F.interpolate(torch.stack([self.terminal[3],self.terminal[4]],2), 6)
		node_features = torch.cat([
								   (head_la[:,:,:3] + head_ra[:,:,:3])/2,
								   torch.stack((lw_ra[:,:,2], lw_ra[:,:,1], lw_ra[:,:,0]), 2),
								   lw_ra[:,:,3:], head_la[:,:,3:], head_ra[:,:,3:]], 2).to(x) # torch.Size([1, 13, 15])
		x = torch.cat((x, node_features.permute(0,2,1).unsqueeze(1).repeat(self.batch_size,32,1,1)), 3) # cat((1 32 15 3), (1 32 15 13))
		
		concat_seq = torch.cat([x[:,:-2], x[:,1:-1], x[:,2:]], 2) # torch.Size([1, 30, 45, 16]) cat((1 30 15 16), (1 30 15 16), (1 30 15 16))
		multi_conv = self.gcl(self.adj, concat_seq)
		_, embedding = self.conv(multi_conv) # 3D: ([1, 32, 15, 16]), 2D: ([1, 32, 15, 15])
		
		return embedding

class AutoEncoderGGCN(GGCN):
	def __init__(self, adj, num_v, num_classes, gc_dims, sc_dims, feat_dims, dropout=0.5, device='cuda', use_2d=True, batch_size=1):
		super(GGCN, self).__init__()
		torch.manual_seed(0)
		terminal_cnt = 5
		actor_cnt = 1
		adj = adj + torch.eye(adj.size(0)).to(adj).detach()
		ident = torch.eye(adj.size(0)).to(adj)
		zeros = torch.zeros(adj.size(0), adj.size(1)).to(adj)
		self.adj = torch.cat([torch.cat([adj, ident, zeros], 1),
							  torch.cat([ident, adj, ident], 1),
							  torch.cat([zeros, ident, adj], 1)], 0).float()
		self.terminal = nn.Parameter(torch.randn(terminal_cnt, actor_cnt, feat_dims))

		self.gcl = GraphConvolution(gc_dims[0]+feat_dims, gc_dims[1], num_v, dropout=dropout)
		self.conv= StandConvolution(sc_dims, num_classes, dropout=dropout)
		self.batch_size = batch_size

		if use_2d:			
			self.decoder = nn.Sequential(
				nn.Linear(192, 320),
				nn.ReLU(),
				nn.Linear(320, 640),
				nn.ReLU(),
				nn.Linear(640, 960),
			).to(device)
		else:
				self.decoder = nn.Sequential(
				nn.Linear(192, 372),
				nn.ReLU(),
				nn.Linear(372, 720),
				nn.ReLU(),
				nn.Linear(720, 1440),
			).to(device)

		nn.init.xavier_normal_(self.terminal)

	def forward(self, x):
		head_la = F.interpolate(torch.stack([self.terminal[0],self.terminal[1]],2), 6)
		head_ra = F.interpolate(torch.stack([self.terminal[0],self.terminal[2]],2), 6)
		lw_ra = F.interpolate(torch.stack([self.terminal[3],self.terminal[4]],2), 6)
		node_features = torch.cat([
								   (head_la[:,:,:3] + head_ra[:,:,:3])/2,
								   torch.stack((lw_ra[:,:,2], lw_ra[:,:,1], lw_ra[:,:,0]), 2),
								   lw_ra[:,:,3:], head_la[:,:,3:], head_ra[:,:,3:]], 2).to(x)
		x = torch.cat((x, node_features.permute(0,2,1).unsqueeze(1).repeat(self.batch_size,32,1,1)), 3)

		concat_seq = torch.cat([x[:,:-2], x[:,1:-1], x[:,2:]], 2) # 1, 30, 45, 3
		multi_conv = self.gcl(self.adj, concat_seq)
		_, embedding = self.conv(multi_conv)
		#breakpoint()
		embedding = nn.Flatten()(embedding)
		x_recons = self.decoder(embedding)
		
		return x_recons, embedding 
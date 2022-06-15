import os

import torch

from stgcn.model import *

class Predictor:
	def __init__(self, model_path, device = None, num_class=61, model_name = "stgcn"):
		if device is None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = device

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
		A = torch.from_numpy(np.asarray(A)).to(self.device)

		num_v = 3
		num_class = num_class
		if model_name == "stgcn":
			self.model = GGCN(A, num_v, num_class, [num_v, num_v*3], [num_v*3, 16, 32, 64], 13, 0.0)
		elif model_name == "stgcn-recons":
			print("load model AutoEncoderGGCN")
			self.model = AutoEncoderGGCN(A, num_v, num_class, [num_v, num_v*3], [num_v*3, 16, 32, 64], 13, 0.0, device)
		else:
			raise NotImplementedError	
		
		if device == 'cuda':
			self.model.cuda()

		self.model.load_state_dict(torch.load(model_path))
		self.model.eval()

		# graph_layer = list(self.model.children())[0]
		# conv_layer = list(self.model.children())[1]
		# conv_layer_wo_fc = torch.nn.Sequential(*(list(conv_layer.children())[:-1]))
		# self.encoder = torch.nn.Sequential(*[graph_layer, conv_layer_wo_fc])
		# self.encoder.eval()
		# self.encoder.cuda()

	def predict(self, x: torch.Tensor):
		"""
		x: 1, 32, 15, 3 or 32, 15, 3
		"""
		if len(x.shape) == 3:
			x = torch.unsqueeze(x, 0)
		x = x.to(self.device)
		with torch.no_grad():
			logit = self.model(x.float())
			predict_label = torch.max(logit, 1)[1]
		return predict_label.cpu().numpy()[0]

	def encode(self, x: torch.Tensor):
		"""
		x: 1, 32, 15, 3 or 32, 15, 3
		"""
		if len(x.shape) == 3:
			x = torch.unsqueeze(x, 0)
		x = x.to(self.device)
		with torch.no_grad():
			embedding = self.model.encode(x.float())
			embedding = torch.flatten(embedding)
		return embedding.cpu().numpy()
import os

import torch

from model import *

class Predictor:
	def __init__(self, model_path, device = None):
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
		num_class = 120
		self.model = GGCN(A, num_v, num_class, [num_v, num_v*3], [num_v*3, 16, 32, 64], 13, 0.0)
		if device == 'cuda':
			self.model.cuda()
		self.model.load_state_dict(torch.load("./models/ntu/model-68.pkl"))
		self.model.eval()

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
				
import random

import torch
import numpy as np
from sklearn.manifold import TSNE
from stgcn.predictor import Predictor as STGCN
import matplotlib.pyplot as plt

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def plot_tsne(model_path, candidates, recons=False):
    if recons:
        predictor = STGCN(model_path, model_name="stgcn-recons")
    else:
        predictor = STGCN(model_path)
    x_labels, y_labels = torch.load("./dataset/ntu_rgb/train.pkl")

    embeddings = []
    labels = []
    print(candidates)
    cdict = {candidates[0]:"magenta",
             candidates[1]:"red",
             candidates[2]:"green",
             candidates[3]:"blue",
             candidates[4]:"yellow"}
    cmap = get_cmap(11)

    for i in range(len(y_labels)):
        if not i%15 == 0:
            continue
        gt_label = y_labels[i].item()
        if gt_label not in candidates:
            continue
        embedding = predictor.encode(x_labels[i])
        embeddings.append(list(embedding))
        labels.append(gt_label)
    embeddings = np.array(embeddings)
    print(embeddings.shape)

    tsne_embedded = TSNE(n_components=2, learning_rate='auto',
                        init='random').fit_transform(embeddings)
    print(tsne_embedded.shape)
    
    plt.cla()
    #plt.title(model_path)
    for embedding, label in zip(tsne_embedded, labels):
        x, y = embedding[0], embedding[1]
        #plt.scatter(x, y, color = cdict[label], s=10)
        plt.scatter(x, y, color=cmap(label), s=10)
    plt.show()

seed = 1234
random.seed = seed
#candidates = random.sample(range(1,61), 5) 
candidates = list(range(1,11)) 
# 8 , 20 or 38
#candidates = [30, 38, 35]
# 원래 가장 잘되던 체리피킹
# candidates = [30, 38, 27, 35]

model_path1 = "./models/ntu_triplet_recons/model-100.pkl"
plot_tsne(model_path1, candidates, recons=True)

model_path2 = "./models/ntu_origin/model-80.pkl"
plot_tsne(model_path2, candidates)



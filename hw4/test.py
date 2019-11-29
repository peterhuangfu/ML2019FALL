import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import model

from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

use_gpu = torch.cuda.is_available()

data_set = np.load(str(sys.argv[1]))
data = np.transpose(data_set, (0, 3, 1, 2)) / 255. * 2 - 1
data = torch.Tensor(data)

auto_encoder = model.AutoEncoder()
auto_encoder.load_state_dict(torch.load('./ae2.pth'))
if use_gpu:
  auto_encoder.cuda()
  data = data.cuda()

test_loader = DataLoader(data, batch_size=60, shuffle=False)

latents = []
reconstructs = []
          
for x in test_loader:
  latent, reconstruct = auto_encoder(x)
  latents.append(latent.cpu().detach().numpy())
  reconstructs.append(reconstruct.cpu().detach().numpy())
    
latents = np.concatenate(latents, axis=0).reshape(9000, -1)
latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)

# latents = PCA(n_components=32, svd_solver='full').fit_transform(latents)
latents = TSNE(n_components=2, random_state=112).fit_transform(latents)
res = KMeans(n_clusters=2, n_jobs=-1, random_state=112).fit(latents).labels_

if np.sum(res[:5]) >= 3:
  res = 1 - res

df = pd.DataFrame({'id': np.arange(0, len(res)), 'label': res})
df.to_csv(str(sys.argv[2]), index=False)

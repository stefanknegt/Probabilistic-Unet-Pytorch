# Probabilistic UNet in PyTorch
A Probabilistic U-Net for segmentation of ambiguous images implemented in PyTorch. This is a pytorch implementation of this paper https://arxiv.org/abs/1806.05034, for which the code can be found here: https://github.com/SimonKohl/probabilistic_unet. 

## Adding KL divergence for Independent distribution
In order to implement an Gaussian distribution with an axis aligned covariance matrix in PyTorch, I needed to wrap a Normal distribution in a Independent distribution. Therefore you need the add the following to the PyTorch source code at torch/distributions/kl.py (source: https://github.com/pytorch/pytorch/issues/13545).

```
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)
```

## Training
In order to train your own Probabilistic UNet in PyTorch, you should first write your own data loader. Then you can use the following code snippet to train the network

```
train_loader = define this yourself
net = ProbabilisticUnet(no_channels,no_classes,filter_list,latent_dim,no_fcomb_convs,beta)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
for epoch in range(epochs):
    for step, (patch, mask) in enumerate(train_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = -elbo + 1e-5 * reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Train on LIDC Dataset
One of the datasets used in the original paper is the [LIDC dataset](https://wiki.cancerimagingarchive.net). I've preprocessed this data and stored them in 5 .pickle files which you can [download here](https://drive.google.com/drive/folders/1xKfKCQo8qa6SAr3u7qWNtQjIphIrvmd5?usp=sharing). After downloading the files you can load the data as follows:
```
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI

dataset = LIDC_IDRI(dataset_location = 'insert_path_here')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
print("Number of training/test patches:", (len(train_indices),len(test_indices)))
```
Combining this with the training code snippet above, you can start training your own Probabilistic UNet.


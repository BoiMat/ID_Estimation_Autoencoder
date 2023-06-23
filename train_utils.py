import numpy as np
import torch
from torch import Tensor
import time
from sklearn.neighbors import NearestNeighbors

def train_epoch(model, device, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = []

    for image_batch, _ in dataloader:

        image_batch = image_batch.to(device)
        decoded_data = model(image_batch)
        loss = loss_fn(decoded_data, image_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def test_epoch(model, device, dataloader, loss_fn):

    model.eval()
    with torch.no_grad():
        conc_out = []
        conc_label = []
        
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            decoded_data = model(image_batch)
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
 
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)

        val_loss = loss_fn(conc_out, conc_label)
        return val_loss.data.item()
        


def save_checkpoint(model, optimizer, num_epochs, filename):
    checkpoint_dict = {
        "parameters": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": num_epochs
    }
    torch.save(checkpoint_dict, filename)
    print("=> Checkpoint saved")

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["parameters"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("=> Checkpoint loaded")


def compute_distances(X, maxk=None):
    maxk = X.shape[0] if maxk is None else maxk
    nbrs = NearestNeighbors(n_neighbors=maxk, algorithm='brute').fit(X)
    distances, _ = nbrs.kneighbors(X)
    print(distances.shape)
    return distances


def compute_id_2NN(X, maxk=5):
    print(f'computiations for {X.shape[0]} datapoints started...')

    start = time.time()
    distances = compute_distances(X)
    end = time.time()
    print(f'distance matrix computed in {end - start: .2f} seconds')

    start = time.time()
    # compute the distance ratios between first and second neighbor
    mus = distances[:, 2] / distances[:, 1]

    # maximum likelihood estimate of the ID
    ndata = X.shape[0]
    id = ndata / np.sum(np.log(mus))

    end = time.time()
    print(f'ID computed in {(end - start) * 1000: .2f} milli seconds\n')

    return id
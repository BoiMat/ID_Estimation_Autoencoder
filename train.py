import torch
from torchvision import datasets, transforms
from conv_AE import MNIST_AE, CIFAR10_AE
from train_utils import train_epoch, test_epoch, load_checkpoint, save_checkpoint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lr = 1e-3
batch_size = 256
num_epochs = 50

trainset = datasets.MNIST(root="datasets/", train=True, transform=transforms.ToTensor(), download=True)
testset = datasets.MNIST(root="datasets/", train=False, transform=transforms.ToTensor(), download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

latent_dims = [8, 32, 64, 256]

for latent_dim in latent_dims:
    
    print('Training MNIST_AE with latent_dim = {}'.format(latent_dim))
    
    model = MNIST_AE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.MSELoss(reduction='sum')

    dict_loss = {'train_loss':[],'val_loss':[]}

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, device, trainloader, loss, optimizer)
        val_loss = test_epoch(model, device, testloader, loss)
        print('EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
        dict_loss['train_loss'].append(train_loss)
        dict_loss['val_loss'].append(val_loss)
        
    save_checkpoint(model, optimizer, num_epochs, 'checkpoints/MNIST_AE_{}.pt'.format(latent_dim))

    with open('losses/MNIST_AE_{}.txt'.format(latent_dim), 'w') as f:
        f.write(str(dict_loss))
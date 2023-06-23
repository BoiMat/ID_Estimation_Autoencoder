import torch

class MNIST_AE(torch.nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1),
            torch.nn.Flatten(),
            torch.nn.Linear(64*6*6, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64*8*8),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, torch.Size([64, 8, 8])),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=2, stride=2, padding=0),
            torch.nn.Sigmoid()      
        )    

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    
    

class CIFAR10_AE(torch.nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(128*7*7, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128*8*8),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, torch.Size([128, 8, 8])),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2),
            torch.nn.Sigmoid()
            
        )    

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
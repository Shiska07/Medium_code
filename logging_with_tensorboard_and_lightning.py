import os
import pdb
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.sampler import SubsetRandomSampler


# helpers
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)

def get_accuracy(logits, targets):

    probabilities = torch.softmax(logits, dim=1)

    # Get the predicted class labels
    predicted_labels = torch.argmax(probabilities, dim=1)

    # Calculate the accuracy
    correct_predictions = (predicted_labels == targets).sum().item()
    total_samples = targets.size(0)
    accuracy = correct_predictions / total_samples

    return accuracy


# datamodule
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = "./", random_seed = 42,
                 val_size=0.2, num_workers=4, pin_memory=True):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.val_size = val_size
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                             ])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # load the dataset
            self.train_dataset = MNIST(root=self.data_dir, train=True,
                                            transform=self.transform)
            self.val_dataset = MNIST(root=self.data_dir, train=True,
                                         transform=self.transform)

            num_train = len(self.train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.val_size * num_train))


            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
            train_idx, val_idx = indices[split:], indices[:split]
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.val_sampler = SubsetRandomSampler(val_idx)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler,
                                               num_workers=self.num_workers, persistent_workers=True, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=self.val_sampler,
                                             num_workers=self.num_workers, persistent_workers=True, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False,
                                              num_workers=self.num_workers, persistent_workers=True, pin_memory=False)


# model
class MnistVAE(pl.LightningModule):
    def __init__(self, input_channels, latent_dim, num_classes):

        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, self.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64 * 7 * 7),
            View((-1, 64, 7, 7)),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1,
                      padding=1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.Linear(256, num_classes)
        )

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu),
                                       torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = F.sigmoid(q.rsample())
        return p, q, z


    def forward(self, x):
        encoder_out = self.encoder(x)
        mu = encoder_out[:, :self.latent_dim]
        logvar = encoder_out[:, self.latent_dim:].clamp(np.log(1e-8), -np.log(
            1e-8))
        p, q, z = self.sample(mu, logvar)

        # get reconstruction
        x_hat = self.decoder(z)

        # get classification
        logits = self.classifier(z)

        return x_hat, logits

    def step(self, batch, batch_idx):

        x, y = batch
        encoder_out = self.encoder(x)
        mu = encoder_out[:, :self.latent_dim]
        logvar = encoder_out[:, self.latent_dim:].clamp(np.log(1e-8), -np.log(
            1e-8))
        p, q, z = self.sample(mu, logvar)

        # get reconstruction
        x_hat = self.decoder(z)

        # get classification
        logits = self.classifier(z)

        # RECONSTRUCTION LOSS
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        # KL-LOSS
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= 0.1

        # CROSS-ENTROPY LOSS
        cross_entropy = torch.nn.functional.cross_entropy(logits, y)

        # ACCURACY
        acc = get_accuracy(logits, y)

        total_loss = recon_loss + kl + cross_entropy

        # store all logging metrics in a dict
        log_dict = {
            "recon_loss": recon_loss,
            "kl_loss": kl,
            "ce_loss": cross_entropy,
            "acc": acc
        }

        return total_loss, log_dict

    def training_step(self, batch, batch_idx):

        loss, logs = self.step(batch, batch_idx)

        # to separate metrics according to stage add tags
        stage_tag = "train"
        for key, val in logs.items():
            self.logger.experiment.add_scalars(key, {stage_tag: val},
                                               self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, logs = self.step(batch, batch_idx)
        stage_tag = "val"
        for key, val in logs.items():
            self.logger.experiment.add_scalars(key, {stage_tag: val},
                                               self.global_step)
        return loss

    def test_step(self, batch, batch_idx):

        loss, logs = self.step(batch, batch_idx)
        stage_tag = "test"

        for key, val in logs.items():
            self.logger.experiment.add_scalars(key, {stage_tag: val}, self.global_step)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-04)


if __name__ == "__main__":
    data_dir = "C:\\Users\\shisk\\Desktop\\Projects\\VAE_thesis\\MNIST\\data"
    batch_size = 64
    random_seed = 42
    val_ratio = 0.2
    num_workers = 4

    # define datamodule
    data_module = MNISTDataModule(batch_size, data_dir, random_seed, val_ratio, num_workers)

    # define model
    input_channels = 1
    latent_dim = 512
    num_classes = 10
    model = MnistVAE(input_channels, latent_dim, num_classes)

    # define logging dir
    logging_dir =  "C:\\Users\\shisk\\Desktop\\Projects\\VAE_thesis\\MNIST" \
                   "\\noteb_logs"
    os.makedirs(logging_dir, exist_ok=True)
    logging_name = "params1"

    # initialize tensorboard logger
    tb_logger = TensorBoardLogger(logging_dir, name=logging_name,
                                  log_graph=False)

    # initialize trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="auto",
        max_epochs=10,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        logger=tb_logger,
        limit_train_batches=200,
        limit_val_batches=1,
        limit_test_batches=1
    )

    # train and test
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

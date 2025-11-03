import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import get_oasis_dataloaders
from modules import UNet, MCDiceLoss


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR = "data/"
    SUBSET_SIZE = -1

    LEARN_RATE = 1e-3

    IN_CHANNELS = 1
    NUM_CLASSES = 4
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    VISUALISE_EVERY = 5


def train_epoch(model, dl, optim, crit, num_classes, dev="cpu"):
    loss = 0

    model.train()
    
    for bat_idx, (imgs, msks) in enumerate(dl):
        print("\tBATCH: %d" % bat_idx)
        imgs = imgs.to(dev)
        msks = msks.to(dev)

        optim.zero_grad()

        logits = model(imgs)

        bat_loss = crit(logits, msks)
        bat_loss.backward()

        optim.step()

        loss += bat_loss.item()

    return loss / len(dl)


def evaluate(model, dl, crit, dev="cpu"):
    loss = 0

    model.eval()

    for bat_idx, (imgs, msks) in enumerate(dl):
        imgs = imgs.to(dev)
        msks = msks.to(dev)

        logits = model(imgs)

        bat_loss = crit(logits, msks).mean()
        loss += bat_loss.item()
    
    return loss / len(dl)


def train(model, dl_train, dl_test, dl_validate, epochs=20, visualise_every=10):
    model.to(Config.DEVICE)

    crit = MCDiceLoss()
    # Use AdaM optimiser
    optim = Adam(model.parameters(), lr=Config.LEARN_RATE)

    train_losses = []

    test_losses = []

    for ep_idx in range(epochs):
        train_losses.append(train_epoch(
            model,
            dl_train,
            optim,
            crit,
            Config.NUM_CLASSES,
            Config.DEVICE
        ))

        test_losses.append(evaluate(
            model, 
            dl_test,
            crit,
            Config.DEVICE
        ))

    return train_losses, test_losses
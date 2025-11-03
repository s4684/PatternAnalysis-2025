import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import get_oasis_dataloaders
from modules import UNet, MCDiceLoss


class Config:
    DATA_DIR = "data/"
    MODEL_SAVE_FILE = "_tdmodel"
    LOSS_SAVE_FILE = "_tdmodel-loss"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SUBSET_SIZE = 100

    LEARN_RATE = 1e-3

    IN_CHANNELS = 1
    NUM_CLASSES = 4
    BATCH_SIZE = 4
    NUM_EPOCHS = 3
    VISUALISE_EVERY = 5


def train_epoch(model, dl, optim, crit, num_classes, dev="cpu"):
    loss = 0

    model.train()
    
    for bat_idx, (imgs, msks) in enumerate(dl):
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


def train(model, dl_train, dl_test, epochs=20, visualise_every=10):
    model.to(Config.DEVICE)
    print("\nStarting model training on %s" % Config.DEVICE)

    crit = MCDiceLoss()
    # Use AdaM optimiser
    optim = Adam(model.parameters(), lr=Config.LEARN_RATE)

    train_losses = []
    test_losses = []

    for ep_idx in range(epochs):
        print("\t[ Epoch %d / %d ]" % (ep_idx + 1, epochs), end="", flush=True)
        
        train_loss = train_epoch(
            model,
            dl_train,
            optim,
            crit,
            Config.NUM_CLASSES,
            Config.DEVICE
        )

        test_loss = evaluate(
            model, 
            dl_test,
            crit,
            Config.DEVICE
        )

        print("\tTrain: %6.5f\tTest: %6.5f" % (train_loss, test_loss))

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    print("Training complete!")

    return train_losses, test_losses


if (__name__ == "__main__"):
    dl_train, dl_test, _ = get_oasis_dataloaders(Config.DATA_DIR, Config.BATCH_SIZE, Config.SUBSET_SIZE)
    model = UNet(Config.IN_CHANNELS, Config.NUM_CLASSES)
    train_loss, test_loss = train(model, dl_train, dl_test, epochs=Config.NUM_EPOCHS)

    torch.save(model.state_dict(), Config.MODEL_SAVE_FILE)
    torch.save([train_loss, test_loss, []], Config.LOSS_SAVE_FILE)
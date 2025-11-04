import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import get_oasis_dataloaders, OasisDataset
from modules import UNet, MCDiceLoss
import matplotlib.pyplot as plt


class Config:
    DATA_DIR = "data/"
    MODEL_SAVE_FILE = "_tdmodel/model.dat"
    LOSS_SAVE_FILE = "_tdmodel/loss.dat"
    DISPLAY_SAVE_PATH = "_tdmodel/disp/"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SUBSET_SIZE = 500

    LEARN_RATE = 1e-3

    IN_CHANNELS = 1
    NUM_CLASSES = 4
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    DISPLAY_EVERY = 2

    display_count = 0


def train_epoch(model, dl, optim, crit, num_classes, dev="cpu", display=False):
    loss = 0

    model.train()
    
    for bat_idx, (imgs, msks) in enumerate(dl):
        imgs = imgs.to(dev)
        msks = msks.to(dev)

        optim.zero_grad()

        logits = model(imgs)

        if (display and bat_idx == 0):
            display_batch(imgs, msks, logits)

        bat_loss = crit(logits, msks)
        bat_loss.backward()

        optim.step()

        loss += bat_loss.item()

    return loss / len(dl)


def evaluate(model, dl, crit, dev="cpu", display=False):
    loss = 0

    model.eval()

    for bat_idx, (imgs, msks) in enumerate(dl):
        imgs = imgs.to(dev)
        msks = msks.to(dev)

        logits = model(imgs)

        if (display and bat_idx == 0):
            display_batch(imgs, msks, logits, save_file="eval.png")

        bat_loss = crit(logits, msks).mean()
        loss += bat_loss.item()
    
    return loss / len(dl)


def display_batch(imgs, msks, logits, save_file=None):

    batch_size = len(imgs)

    fig, axes = plt.subplots(batch_size, 3)
    axes[0][0].set_title("Input Image")
    axes[0][1].set_title("Ground Truth")
    axes[0][2].set_title("Prediction")

    imgs_data = imgs.cpu().numpy()
    msks_data = msks.cpu().numpy()
    logits_data = logits.argmax(dim=1).cpu().numpy()

    with torch.no_grad():
        for bat_idx in range(batch_size):
            img_data = imgs_data[bat_idx, 0]
            msk_data = msks_data[bat_idx]
            logit_data = logits_data[bat_idx]

            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)

            msk_data = OasisDataset.encode_mask(msk_data)
            logit_data = OasisDataset.encode_mask(logit_data)

            axes[bat_idx][0].imshow(img_data, cmap="gray")
            axes[bat_idx][1].imshow(msk_data, cmap="gray")
            axes[bat_idx][2].imshow(logit_data, cmap="gray")

    plt.tight_layout()

    if (save_file == None):
        save_file = "%d.png" % Config.display_count
    
    plt.savefig(Config.DISPLAY_SAVE_PATH + save_file)
    plt.close()
    
    Config.display_count += 1


def train(model, dl_train, dl_test, epochs=20, display_every=10):
    model.to(Config.DEVICE)
    print("\nStarting model training on %s" % Config.DEVICE)

    crit = MCDiceLoss()
    # Use AdaM optimiser
    optim = Adam(model.parameters(), lr=Config.LEARN_RATE)

    train_losses = []
    test_losses = []

    for ep_idx in range(1, epochs + 1):
        print("\t[ Epoch %d / %d ]" % (ep_idx, epochs), end="", flush=True)
        
        train_loss = train_epoch(
            model,
            dl_train,
            optim,
            crit,
            Config.NUM_CLASSES,
            Config.DEVICE,
            not bool(ep_idx % display_every)
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
    train_loss, test_loss = train(model, dl_train, dl_test, epochs=Config.NUM_EPOCHS, display_every=Config.DISPLAY_EVERY)

    torch.save(model.state_dict(), Config.MODEL_SAVE_FILE)
    torch.save([train_loss, test_loss, []], Config.LOSS_SAVE_FILE)
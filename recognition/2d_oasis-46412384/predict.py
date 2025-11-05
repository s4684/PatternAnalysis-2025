import torch
import torch.nn as nn
from train import Config, evaluate
from dataset import get_oasis_dataloaders
from modules import UNet, MCDiceLoss
import matplotlib.pyplot as plt


def plot_loss(train_loss, validate_loss, save_path, title="Loss over Training Epochs", legend=["Training loss", "Validation loss"]):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.title(title)

    plt.plot(train_loss, "r")
    plt.plot(validate_loss, "b")

    plt.legend(legend)

    plt.savefig(save_path)
    plt.close()


if (__name__ == "__main__"):
    _, _, dl_test = get_oasis_dataloaders(Config.DATA_DIR, Config.BATCH_SIZE, Config.SUBSET_SIZE)
    model = UNet(Config.IN_CHANNELS, Config.NUM_CLASSES).to(Config.DEVICE)

    # Read model data from save file
    model.load_state_dict(torch.load(Config.MODEL_SAVE_FILE, map_location=Config.DEVICE))
    train_loss, train_dice, validate_loss, validate_dice = torch.load(Config.LOSS_SAVE_FILE)

    print("\nStarting model evaluation")

    crit = nn.CrossEntropyLoss()
    dice_fn = MCDiceLoss()

    print("\t[ Eval ]", end="", flush=True)

    # """
    test_loss, test_dice = evaluate(
        model=model, 
        dl=dl_test, 
        crit=crit, 
        dice_fn=dice_fn,
        dev=Config.DEVICE,
        display=True
    )
    # """

    plot_loss(
        train_loss, 
        validate_loss, 
        save_path=Config.PLOT_SAVE_PATH + "celoss.png", 
        title="Cross-Entropy Loss over Training Epochs",
        legend=["Training CE Loss", "Validation CE Loss"]
    )

    plot_loss(
        train_dice, 
        validate_dice, 
        save_path=Config.PLOT_SAVE_PATH + "diceloss.png", 
        title="Dice Loss over Training Epochs",
        legend=["Training Dice Loss", "Validation Dice Loss"]
    )
 
    print("\tValidate: L=%6.5f D=%6.5f" % (test_loss, test_dice))

    print("Evaluation complete!")
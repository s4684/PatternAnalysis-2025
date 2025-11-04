import torch
from train import Config, evaluate
from dataset import get_oasis_dataloaders
from modules import UNet, MCDiceLoss


if (__name__ == "__main__"):
    _, _, dl_test = get_oasis_dataloaders(Config.DATA_DIR, Config.BATCH_SIZE, Config.SUBSET_SIZE)
    model = UNet(Config.IN_CHANNELS, Config.NUM_CLASSES).to(Config.DEVICE)

    # Read model data from save file
    model.load_state_dict(torch.load(Config.MODEL_SAVE_FILE, map_location=Config.DEVICE))
    train_losses, validate_losses, test_losses = torch.load(Config.LOSS_SAVE_FILE)

    print("\nStarting model evaluation")

    crit = MCDiceLoss()

    test_loss = evaluate(
        model, 
        dl_test, 
        crit, 
        Config.DEVICE,
        display=True
    )
 
    print("\t[ Eval ]\tValidate: %6.5f" % test_loss)
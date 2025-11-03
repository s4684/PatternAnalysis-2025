import torch
from train import Config, evaluate
from dataset import get_oasis_dataloaders
from modules import UNet, MCDiceLoss


if (__name__ == "__main__"):
    _, _, dl_validate = get_oasis_dataloaders(Config.DATA_DIR, Config.BATCH_SIZE, Config.SUBSET_SIZE)
    model = UNet(Config.IN_CHANNELS, Config.NUM_CLASSES).to(Config.DEVICE)

    # Read model data from save file
    model.load_state_dict(torch.load(Config.MODEL_SAVE_FILE, map_location=Config.DEVICE))
    train_losses, test_losses, validate_losses = torch.load(Config.LOSS_SAVE_FILE)

    print("\nStarting model evaluation")

    crit = MCDiceLoss()

    validate_loss = evaluate(
        model, 
        dl_validate, 
        crit, 
        Config.DEVICE
    )
 
    print("\t[ Eval ]\tValidate: %6.5f" % validate_loss)
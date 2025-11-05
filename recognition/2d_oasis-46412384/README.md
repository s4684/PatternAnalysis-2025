# 2D OASIS brain data segmentation with Improved UNet

## Project Synopsis

This project segments raw 2D brain MRI data slices into regions of distinct classification. 
Using an improved UNet model, success criteria consisted of achieving a dice similarity coefficient of 0.9 minimum across the final predictions.\
Each 2D data slice is segmented into 4 classes, distinguished by equally distributed greyscale shades in the segmentation mask:

1. Background (black)
2. Cerebro-spinal fluid (dark grey)
3. Grey matter (light grey)
4. White matter (white)

For example:

![example batch prediction visualisation](ex_figs/ex-batch_preds.png)


## Model Analysis

### UNet Model

The Improved UNet algorithm proves to be a reliable model architecture for 2D image segmentation tasks, particularly those of medical application. 
In summary, it's design revolves around applying numerous fine filters over the image to discern patterns across the image. 
These filters are applied repeatedly over several layers, between which the data is compressed, yielding a smaller result each time.
This act of compression - called pooling - allows for the algorithm to capture elements of both fine detail and abstract patterns across the data.
Finally, the data is upscaled back through each layer, where each layer of prediction is compiled presenting a final mask that matches the resolution of the input image.
This method of compaction to a final bottleneck and contrasting reconstruction presents the U-shape architecture that entitles the UNet.

### Dataset

This model is trained on the pre-split OASIS dataset. Training comprised 9664 slices, validation over 1120 validation slices, and final evaluation on 544 testing slices.
This achieves a distribution approximating 85 / 10 / 5 respectively, aligning with the train-heavier side of typical split ratios.

The dataset consistutes 256x256 greyscale (single channel) images. Loading of each dataset consisted of:

- Sequentially pairing each slice with their respective mask
- Applying z-score normalization to each slice, minimising instability in training values
- Decoding the masks from greyscale values to class-label identifiers (integers 0 - 3)

Such processing was computed on-the-fly, as data was requested by the model. 
This increases the training time, as data was processed repeatedly upon each epoch, in favour of optimising the memory demands.

### Model Operation

Training consisted of 26 epochs and batches of 4 images, with validation tests occurring after each epoch. 
Model evaluation utilised a combination of Dice loss coefficient and cross entropy loss methods calculated at each epoch, however testing was governed by cross entropy loss.
Train the model using:
```
python3 train.py
```
Each epoch loss will be displayed, and the model state and training losses are saved to disk (model: `_tdmodel/model.dat`; losses: `_tdmodel/loss.dat`).\
Snapshots of the predictions throughout training are also saved locally (`_tdmodel/disp/`)

### Model Evaluation

After training, evaluate the model using:
```
python3 predict.py
```
This will load the saved model and test it using the yet unseen test dataset. 
Its final result is displayed in output and the visualisation of predictions in the first batch is saved to disk.\
For example:

![example evaluation batch visualisation](ex_figs/ex-eval_preds.png)

This module will also model the loss results from training and validation (`_tdmodel/plot/`).\
For example:

![example cross-entropy loss plot](ex_figs/ex-celoss.png)


## Dependencies

- `python : 3.9.23`
- `pytorch : 2.5.1`
- `numpy : 2.0.1`
- `pillow : 11.3.0`
- `matplotlib : 3.9.2`
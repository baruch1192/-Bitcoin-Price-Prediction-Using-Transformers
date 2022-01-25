## <center>Bitcoin Price Prediction Using Transformers</center>

In this project we used Transformers architecture - encoder-decoder, to predict Bitcoin value into a chosen future horizon. 
Our raw data holds almost 1 year of Bitcoin prices per minute (closing, opening, etc.). We extracted more statistics out of the data using common financial technical indicators - Finta, while making sure they are low correlated between them. We then fed the model with the data and trained it to predict the chosen future horizon based on past values.
We optimized the hyperparameters of the model using Optuna.

Model's prediction on the test set:
![alt text](https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Test_Prediction.png)

- [Bitcoin Price Prediction Using Transformers](#Bitcoin-Price-Prediction-Using-Transformers)
  * [Previous Work](#Previous-Work)
  * [Data Proccessing](#Data-Proccessing)
  * [Architecture](#Architecture)
  * [Parameters](#Parameters)
  * [Optuna](#Optuna)
  * [Results](#Results)
  * [Usage](#Usage)
  * [Files in the Repository](#Files-in-the-Repository)
  * [Further Work](#Further-Work)
  * [DISCLAIMER](#DISCLAIMER)



## Previous Work
Bitcoin prediction using RNN:

&emsp; https://www.kaggle.com/muharremyasar/btc-historical-with-rnn

IBM stock price prediction using Transformer-Encoder:

&emsp; https://github.com/JanSchm/CapMarket/blob/master/bot_experiments/IBM_Transformer%2BTimeEmbedding.ipynb

Short term stock price prediction using LSTM with a simple trading bot:

&emsp; https://github.com/roeeben/Stock-Price-Prediction-With-a-Bot/blob/main/README.md



## Data Processing
We are using Bitcoin historical one-minute records from (UTC+8): 2021-01-01 00:00:00 - 2021-12-05 23:59:00, containing 488,160 records from Okex Exchange.
We got it from: https://www.kaggle.com/aipeli/btcusdt and it can also be found in our repository [here](https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/data/okex_btcusdt_kline_1m.csv.zip).

The data contains 5 features: the opening price, highest price, lowest price, closing price, and volume of transactions per minute.
We calculated the correlations between the features and noticed that the first 4 (the prices) are high-correlated between themselves. So we wanted to add more meaningful features to the data before handing it to the model. For that, we used [FinTA](https://github.com/peerchemist/finta) which implements common financial technical indicators in Pandas. We chose only the features which are low-correlated to all others and made sure they all use only past samples (so we won't accidentally use the future). After choosing them we cleaned it from NaNs and ended up with a total of 34 features and 488029 samples (lost the first 131 samples).

After that we splitted the data into train (80%), validation (10%) and test (10%), in chronological order as can be seen here:

![alt text](https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Data_Separation.png)

Then the train data is being scaled, and the validation and test datasets are scaled accordingly. 

Finally, we divided the train set into tensors of large sequential batches. During the training, we will sample from each batch a sequence of `bptt_src` to use as source and a sequence of `bptt_tgt` to use as target. To create more diverse data, we can start sample from a random start point in each epoch, by setting the flag `random_start_point` True.


## Architecture
We used PyTorch nn.Transformer as the basis of our model. Before both encoder and decoder we entered time embedding layer and in the output of the decoder a linear one.

In the time embedding layer we are implementing a version of [Time2Vec](https://arxiv.org/pdf/1907.05321.pdf). We added more features to the data in 2 ways:
1. Periodic features implemented as a linear layer followed by sin activation - total of `periodic_features` features.
2. Linear features implemented as a linear layer.

Both kinds of features are concatanated to the existing ones creating a total of `out_features` at the output.

The linear layer before the output is used to output the same number of features as the target - `in_features`.

The model structure:
<p align="center">
  <img src="https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Model_Structure.png" width="450"/>
</p>


# Parameters
* `num_features` = int, number of features to choose from the full set (1 - 34)
* `scaler` = str, the kind of scaler to use to scale the data (Standard Scaler - 'standard', Min Max Scaler - 'minmax')
* `train_batch_size` = int, size of train batch
* `eval_batch_size` = int, size of validation/test batch 
* `epochs` = int, number of epochs to run the training
* `bptt_src` = int, the length of the source sequence
* `bptt_tgt` = int, the length of the target sequence
* `overlap` = int, number of overlapping samples between the source and the target
* `num_encoder_layers` = int, number of enconder layers in the transformer 
* `num_decoder_layers` = int, number of decoder layers in the transformer 
* `periodic_features` = int, number of periodic features to add in the time embedding layer
* `out_features` = int, number of output feature after the time embedding layer (> `in_features + periodic_features`)
* `nhead` = int, number of heads in the multihead attention layers in the transformer (both encoder and decoder, must be a divider of `out_features`)
* `dim_feedforward` = int, dimension of the feed forward layers in the transformer (both encoder and decoder)
* `dropout` = float, the dropout probability of the dropout layers in the model (0.0 - 1.0)
* `activation` = str, activation function to use in the transformer (ReLU - 'relu', GeLU - 'gelu')
* `random_start_point` = bool, start each epoch from random start point from the first `bptt_src` samples
* `clip_param` = float, the max norm of the gradients in the clip_grad_norm layer 
* `lr` = float, starting learning rate 
* `gamma` = float, multiplicative factor of learning rate decay (0.0 - 1.0)
* `step_size` = int, period of learning rate decay in epochs

The most crucial thing to understand here is the relations between `bptt_src`, `bptt_tgt` and `overlap`. We use `bptt_src` past samples to predict the following `bptt_tgt - overlap`.

## Optuna
We used Optuna in order to find the optimal hyperparameters in terms of the validation loss.

We fixed or chose a deterministic function to some of the hyperparameters by using the knowledge we gained during the manual tuning, to make runtime more reasonable:

|Hyperparameter   | Value |
|-------------|------|
|`num_features`| 34 |
|`train_batch_size`| 32 |
|`eval_batch_size `| 32 |
|`epochs `| 50 |
|`overlap `| 1 |
|`num_decoder_layers`| `num_encoder_layers` |
|`periodic_features`| (`out_features` - `num_features` // 10) * 4 + 2 |
|`nhead`| `out_features` / 4|
|`step_size `| 1 |
|`lr `| 0.5 |
|`step_size `| 1 |
|`gamma `| 0.95 |

For the other hyperparameters we chose the range of possible values to optimize over.

These are the hyperparameter that was chosen:
 
|Hyperparameter   | Value |
|-------------|------|
|`scaler`| 'minmax' |
|`bptt_src`| 10 |
|`bptt_tgt`| 6 |
|`num_encoder_layers`| 4 |
|`out_features`| 60 |
|`dim_feedforward`| 384 |
|`dropout`| 0.0 |
|`random_start_point`| 'False' |
|`clip_param`| 0.75 |


The impact of these hyperparameters on the loss is visualized here: 

<p align="center">
  <img src="https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Optuna_Result.jpeg" />
</p>

The most important one is the `scaler`.
We also saw that `bptt_src` and `bptt_tgt` were not that important as we thought they would be.

After our final fine tuning we only changed `bptt_tgt` from 6 as suggested by optuna to 2.

The full analysis by Optuna can be found in [bitcoin_price_prediction_optuna.ipynb](https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/data/bitcoin_price_prediction_optuna.ipynb)


## Results

We trained the model with the hyperparameters above. 

Model statistics:
* Train Loss: <img src="https://render.githubusercontent.com/render/math?math=2.4\cdot10^{-5}">
* Validation Loss: <img src="https://render.githubusercontent.com/render/math?math=6.1\cdot10^{-5}">
* Test Loss: <img src="https://render.githubusercontent.com/render/math?math=10.1\cdot10^{-5}">

After this training we checked the real-time performence of the model on the test set, meaning we entered the first 10 samples as source (`bptt_src` = 10), the 10th sample in this sequence as target (`overlap` = 1) predicted the next value (`bptt_tgt` - `overlap` = 1). We then shifted the source samples by one and predicted the next value in the same way. We repeated the process until we had the prediction for all the possible minutes in the test set. You can see the result here:

<p align="center">
  <img src="https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Test_Prediction.png" />
</p>

We can see that the general trend of the prediction is similar to the real one. That is not surprising because we are looking on a large scale of minutes, 48,802, where we the prediction is only based on the last 10 samples, so in the large scale we expect to see both real and prediction around the same values. For better analysis we need to look closer, so here is zoom-in view:

<p align="center">
  <img src="https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Test_Presiction_Zoom_In.png" />
</p>

Here we can see the differences between the real and predicted values. The trends are still somewhat similiar but sometimes the predidtion predicts a rise or fall before it happens, for example the rise around the minute 42,560 or the little fall in minute 42,580, and sometimes it just follows the existing trend like around minute 42,600.


# Usage

To retrain the model run [bitcoin_price_prediction.ipynb](https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/data/bitcoin_price_prediction.ipynb) after you chose your hyperparameters in the first cell. The flag `plot_data_process` when set False will hide all the produced data processing image.

If you would like to do further hyperparameters tuning using optuna run [bitcoin_price_prediction_optuna.ipynb](https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/data/bitcoin_price_prediction_optuna.ipynb). In the `define_model` function we decalred the fixed hyperparameters values and in `objective` function we declared the hyperparameters we want to tune along woth their range. 

First, anyone who'd like is welcome to just run the `Model Training - CGEN.ipynb` to retrain the model, or run `stock_bot.py` to run the bot and/or edit it.

# Parameters
* kernel_size = size of the line segement kernel (usually 1/30 of the height/width of the original image)
* stroke_width = thickness of the strokes in the Stroke Map (0, 1, 2)
* num_of_directions = stroke directions in the Stroke Map (used for the kernels)
* smooth_kernel = how the image is smoothed (Gaussian Kernel - "gauss", Median Filter - "median")
* gradient_method = how the gradients for the Stroke Map are calculated (0 - forward gradient, 1 - Sobel)
* rgb = True if the original image has 3 channels, False if grayscale
* w_group = 3 possible weight groups (0, 1, 2) for the histogram distribution, according to the paper (brighter to darker)
* pencil_texture_path = path to the Pencil Texture Map to use (4 options in "./pencils", you can add your own)
* stroke_darkness = 1 is the same, up is darker.
* tone_darkness = as above

# Folders
* inputs: test images from the publishers' website: http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/pencil_drawing.htm
* pencils: pencil textures for generating the Pencil Texture Map

# Reference
[1] Lu C, Xu L, Jia J. Combining sketch and tone for pencil drawing production[C]//Proceedings of the Symposium on Non-Photorealistic Animation and Rendering. Eurographics Association, 2012: 65-73.

[2] Matlab implementation by "candtcat1992" - https://github.com/candycat1992/PencilDrawing



## Files in the repository

| Folder |File name         | Purpose |
|------|----------------------|------|
|`data_prep.py`| All of the data transformations (train/valid/test splits & feature engineering) |
|`model.py`| The LSTM model|
|`stock_bot.py`| Everything that's related to the bot, including to its definition and simulations |
|`train.py`| The training loop|
|`CGEN_original.pkl`| The actual test prices for the bot to use |
|`CGEN_predict.pkl`| Our prediction for the test prices for the bot to use |
|`Model Training - CGEN.ipynb`| A notebook which shows our data with the features, aswell as the training procedure and graphs |
|`Optuna Optimization - CGEN.ipynb`| A notebook which has the entire hyperparameters optimization using Optuna|




## Further Work

First, anyone who'd like is welcome to just run the `Model Training - CGEN.ipynb` to retrain the model, or run `stock_bot.py` to run the bot and/or edit it.

We most definitely have some improvements to our small project in mind, including but not limited to:
- Involving various stocks instead of a single one.
- Calculating many more features (using [FinTA](https://github.com/peerchemist/finta)) and have [Optuna](https://github.com/optuna/optuna) to choose which features to pick. 
- Making the bot more sophisticated, maybe not deterministic or actually train its hyperparameters on a validation set or another stock.
- Training for a longer period: we specifically cut a period in which the Buy&Hold strategy loses a bit because we wanted to compete it, but it doesn't have to be the case.


## References
* [A nice LSTM article](https://web.stanford.edu/class/cs379c/archive/2018/class_messages_listing/content/Artificial_Neural_Network_Technology_Tutorials/OlahLSTM-NEURAL-NETWORK-TUTORIAL-15.pdf) by Stanford.
* [FinTA](https://github.com/peerchemist/finta).
* [Optuna](https://github.com/optuna/optuna).


## DISCLAIMER

We'd like to further emphasize in addition to our comments above that our predictions and bot are nowhere near reliable and we'd highly advise not to try any of the provided here in a live or real setting.

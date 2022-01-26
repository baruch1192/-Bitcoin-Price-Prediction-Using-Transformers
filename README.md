# <center>Bitcoin Price Prediction Using Transformers</center>

In this project we used Transformers architecture - encoder-decoder, to predict Bitcoin value into a chosen future horizon. 
Our raw data holds almost 1 year of Bitcoin prices per minute (closing, opening, etc.). We extracted more statistics out of the data using common financial technical indicators - Finta, while making sure they are low correlated between them. We then fed the model with the data and trained it to predict the chosen future horizon based on past values.
We optimized the hyperparameters of the model using Optuna.

<p align="center">
  <img src='https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/presentation_preview.gif' width=600 ></a>
</p>

<p align="center">
Watch on Youtube:  <a href="https://youtu.be/tb_47ng7ZOI"><img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=20></a>
</p>
 
- [Bitcoin Price Prediction Using Transformers](#Bitcoin-Price-Prediction-Using-Transformers)
  * [Previous Work](#Previous-Work)
  * [Data Processing](#Data-Processing)
  * [Architecture](#Architecture)
  * [Hyperparameters](#Hyperparameters)
  * [Optuna](#Optuna)
  * [Result](#Result)
  * [Usage](#Usage)
  * [Files in the Repository](#Files-in-the-Repository)
  * [Further Work](#Further-Work)


## Previous Work
Bitcoin prediction using RNN:

https://www.kaggle.com/muharremyasar/btc-historical-with-rnn

IBM stock price prediction using Transformer-Encoder:

https://github.com/JanSchm/CapMarket/blob/master/bot_experiments/IBM_Transformer%2BTimeEmbedding.ipynb

Short term stock price prediction using LSTM with a simple trading bot:

https://github.com/roeeben/Stock-Price-Prediction-With-a-Bot/blob/main/README.md



## Data Processing
We are using Bitcoin historical one-minute records from (UTC+8): 2021-01-01 00:00:00 - 2021-12-05 23:59:00, containing 488,160 records from Okex Exchange.
We got it from: https://www.kaggle.com/aipeli/btcusdt and it can also be found in our repository [here](https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/data/okex_btcusdt_kline_1m.csv.zip).

Without regard to the time stamp feature, the data contains 5 features: the opening price, highest price, lowest price, closing price, and volume of transactions per minute.
We calculated the correlations between the features and noticed that the first 4 (the prices) are high-correlated between themselves. So we wanted to add more meaningful features to the data before handing it to the model. For that, we used [FinTA](https://github.com/peerchemist/finta) which implements common financial technical indicators in Pandas. We chose only the features which are low-correlated to all others and made sure they all use only past samples (so we won't accidentally use the future). After choosing them we cleaned it from NaNs and ended up with a total of 34 features and 488029 samples (lost the first 131 samples).

After that we split the data into train (80%), validation (10%), and test (10%), in chronological order as can be seen here:

![alt text](https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Data_Separation.png)

Then the train data is being scaled, and the validation and test datasets are scaled accordingly. 

Finally, we divided the train set into tensors of large sequential batches. During the training, we will sample from each batch a sequence of `bptt_src` to use as source and a sequence of `bptt_tgt` to use as target. To create more diverse data, we can start to sample from a random start point in each epoch, by setting the flag `random_start_point` True.


## Architecture
We used PyTorch nn.Transformer as the basis of our model. Before both encoder and decoder, we entered a time embedding layer and in the output of the decoder a linear one.

In the time embedding layer, we are implementing a version of [Time2Vec](https://arxiv.org/pdf/1907.05321.pdf). We added more features to the data in 2 ways:
1. Periodic features which  implemented as a linear layer followed by sin activation - a total of `periodic_features` features.
2. Linear features which implemented as a linear layer.

Both kinds of features are concatenated to the existing ones creating a total of `out_features` at the output.

The linear layer before the output is used to output the same number of features as the target - `in_features`.

The model structure:
<p align="center">
  <img src="https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Model_Structure.png" width="450"/>
</p>


## Hyperparameters
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
We used Optuna to find the optimal hyperparameters in terms of the validation loss.

We fixed or constrained some of the hyperparameters by using the knowledge we gained during the manual tuning, to make runtime more reasonable:

|Hyperparameter   | Value |
|-------------|------|
|`num_features`| 34 |
|`train_batch_size`| 32 |
|`eval_batch_size `| 32 |
|`epochs `| 50 |
|`overlap `| 1 |
|`num_decoder_layers`| `num_encoder_layers` |
|`periodic_features`| (`out_features` - `num_features` // 10) x 4 + 2 |
|`nhead`| `out_features` / 4|
|`step_size `| 1 |
|`lr `| 0.5 |
|`step_size `| 1 |
|`gamma `| 0.95 |

For the other hyperparameters, we chose the range of possible values to optimize over.

These are the hyperparameters that were chosen:
 
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
We also saw that `bptt_src` and `bptt_tgt` were not as important as we thought they would be.

After our final fine-tuning, we only changed `bptt_tgt` from 6 as suggested by optuna to 2.

The full analysis by Optuna can be found in [bitcoin_price_prediction_optuna.ipynb](https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/data/bitcoin_price_prediction_optuna.ipynb)


## Result

We trained the model with the hyperparameters above. 

Model statistics:
* Train Loss: <img src="https://render.githubusercontent.com/render/math?math=2.4\cdot10^{-5}">
* Validation Loss: <img src="https://render.githubusercontent.com/render/math?math=6.1\cdot10^{-5}">
* Test Loss: <img src="https://render.githubusercontent.com/render/math?math=10.1\cdot10^{-5}">

After this training we checked the real-time performance of the model on the test set, meaning we entered the first 10 samples as source (`bptt_src` = 10), the 10th sample in this sequence as target (`overlap` = 1) and predicted the next value (`bptt_tgt` - `overlap` = 1). We then shifted the source samples by one and predicted the next value in the same way. We repeated the process until we had the prediction for all the possible minutes in the test set. You can see the result here:

<p align="center">
  <img src="https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Test_Prediction.png" />
</p>

We can see that the general trend of the prediction is similar to the real one. That is not surprising because we are looking at a large scale of minutes, 48,802, where the prediction is only based on the last 10 samples, so on the large scale, we expect to see both real and prediction around the same values. For better analysis we need to look closer, so here is a zoom-in view:

<p align="center">
  <img src="https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Test_Presiction_Zoom_In.png" />
</p>

Here we can see the differences between the real and predicted values. The trends are still somewhat similar but sometimes the prediction predicts a rise or fall before it happens, for example, the rise around the minute 42,560 or the little fall in minute 42,580, and sometimes it just follows the existing trend like around minute 42,600.


## Usage

To retrain the model run [bitcoin_price_prediction.ipynb](https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/data/bitcoin_price_prediction.ipynb) after you chose your hyperparameters in the first cell. The flag `plot_data_process` when set False will hide all the produced data processing images.

If you would like to do further hyperparameters tuning using optuna run [bitcoin_price_prediction_optuna.ipynb](https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/data/bitcoin_price_prediction_optuna.ipynb). In the `define_model` function we declared the values of the fixed or constrained hyperparameters and in the `objective` function we declared the hyperparameters we want to tune along with their range.



## Files in the repository

| Folder |File name         | Purpose |
|------|----------------------|------|
|code|`bitcoin_price_prediction.ipynb`| Notebook which includes all data processing, training, and inference |
| |`bitcoin_price_prediction_optuna.ipynb`| Optuna hyperparameters tuning |
|data|`okex_btcusdt_kline_1m.csv.zip`| Zip file containing the data we used in this project |
|images|`Data_Separation.png`| Image that shows our train-validation-tets split |
| |`Model_Structure.png`| Image that shows our model architecture |
| |`Optuna_Result.jpeg`| Image that shows the importance of the Hyperparameters produced by Optuna  |
| |`Test_Prediction.png`| Image that shows our result on the test set |
| |`Test_Presiction_Zoom_In.png`| Image that shows our result on the test set - zoomed-in|
| |`presentation_preview.gif`| Gif showing preview of the project presentation|


## Further Work

The work we presented here achieved good results, but definitely there are aspects to improve and examine such as:
- Try running the model on a different stock.
- Examine the feature extraction process and check which features are the most helpful.
- Further tuning of the hyperparameters, release the constraints we put on some of them.
- Check the performance in real-time trading (better to start with a trading bot on the test set)

Hope this was helpful and please let us know if you have any comments on this work:

https://github.com/yuvalaya

https://github.com/baruch1192

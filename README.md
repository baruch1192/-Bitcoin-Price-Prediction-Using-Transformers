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
* `num_features` = int, number of features to chose from the full set (1 - 34)
* `scaler` = str, the kind of scaler to use to scale the data (Standard Scaler - 'standard', Min Max Scaler - 'minmax')
* `train_batch_size` = int, size of train batch
* `eval_batch_size` = int, size of validation/test batch 
* `epochs` = int, number of epochs to run the training
* `bptt_src` = int, the length of the source sequence
* `bptt_tgt` = int, the length of the target sequence
* `overlap` = number of overlapping samples between the source and the target
* `num_encoder_layers` = number of enconder layers in the transformer 
* `num_decoder_layers` = number of decoder layers in the transformer 
* `periodic_features` = number of periodic features to add in the time embedding layer
* `out_features` = number of output feature after the time embedding layer (> in_features + periodic_features)
* `nhead` = number of heads in the multihead attention layers in the transformer (both encoder and decoder, must be a divider of out_features)
* `dim_feedforward` = dimension of the feed forward layers in the transformer (both encoder and decoder)
* `dropout` = the dropout probability of the dropout layers in the model
* `activation` = activation function to use in the transformer (ReLU - 'relu', GeLU - 'gelu')
* `random_start_point` = bool, start each epoch from random start point in range of bptt_src
* `clip_param` = the max norm of the gradients in the clip_grad_norm layer
* `lr` = starting learning rate 
* `gamma` = multiplicative factor of learning rate decay
* `step_size` =  period of learning rate decay in epochs

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

These are the hyperparameter the was chosen:
 
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
Statistics:
* Train Loss: <img src="https://render.githubusercontent.com/render/math?math=2.4\cdot10^{-5}">
* Validation Loss: 6.1 \cdot 10^-5
* Test Loss: 10.1 \cdot 10^-5

After training the model for the mentioned period, feeding the test data to get a prediction and then giving the actual & predicted prices to the bot, we got the following:

<p class="aligncenter">
<img src="./assets/CGEN test.png">
</p>

In green we see the points in which the model buys, and in red: sells. 
The bot has had 52% of successful trades, but we don't give this number too much thoughts since a trader can have a profit even with 10% successful trades, aslong as the losing trades don't lose as much as the winning trades gain.
Overall, it was given 5000$ at the start of the simulation and finished with a value of 5204$, i.e. it gained 200$, which the Buy&Holder has lost 54$.

Up top we have the animation of the bot running in action on the test data, and if we freeze the animation at the end:

<p class="aligncenter">
  <img src="./assets/Bot on Test.png">
</p>

We can see how the bot made its profit: from minute ~600 to ~1100 we have an increase of the stock price, which the bot manages to utilize, and after that the stock starts to decrease back. While the Buy&Hold strategy lose all the previously gained money in that period, it seems that our bot manages to detect the fall and sells everything, and thus hold on to its profits.

We are, however, aware that the bot wasn't tested on a long enough period, and this method is very much likely to fail in a more diverse setting. What we're showing is a success for this specific period (even though there weren't any tuning on that period, since it's the test set).




https://towardsdatascience.com/stock-predictions-with-state-of-the-art-transformer-and-time-embeddings-3a4485237de6


The model was written in a general fashion: we set all of his layers size, aswell as the mentioned `bptt` and `batch_size` as variables and proceeded to use [Optuna](https://github.com/optuna/optuna) to optimize over the validation loss.

This loss, aswell as the training loss, was defined as the MSE between the LSTM's prediction of a minute and the closing price of the next minute, which is of course what we're trying to predict.

### Based on the paper "Combining Sketch and Tone for Pencil Drawing Production" by Cewu Lu, Li Xu, Jiaya Jia
#### International Symposium on Non-Photorealistic Animation and Rendering (NPAR 2012), June 2012
Project site can be found here:
http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/pencil_drawing.htm

Paper PDF - http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/npar12_pencil.pdf

Draws inspiration from the Matlab implementation by "candtcat1992" - https://github.com/candycat1992/PencilDrawing

In this notebook, we will explain and implement the algorithm described in the paper. This is what we are trying to achieve:
![alt text](https://github.com/taldatech/image2pencil-drawing/blob/master/images/ExampleResult.JPG)

We can divide the workflow into 2 main steps:
1. Pencil stroke generation (captures the general strucure of the scene)
2. Pencil tone drawing (captures shapes shadows and shading)

Combining the results from these steps should yield the desired result. The workflow can be depicted as follows:
![alt text](https://github.com/taldatech/image2pencil-drawing/blob/master/images/Workflow.JPG)

* Both figures were taken from the original paper

Another example:
![alt text](https://github.com/taldatech/image2pencil-drawing/blob/master/images/jl_compare.JPG)

# Usage
```python
from PencilDrawingBySketchAndTone import *
import matplotlib.pyplot as plt
ex_img = io.imread('./inputs/11--128.jpg')
pencil_tex = './pencils/pencil1.jpg'
ex_im_pen = gen_pencil_drawing(ex_img, kernel_size=8, stroke_width=0, num_of_directions=8, smooth_kernel="gauss",
                       gradient_method=0, rgb=True, w_group=2, pencil_texture_path=pencil_tex,
                       stroke_darkness= 2,tone_darkness=1.5)
plt.rcParams['figure.figsize'] = [16,10]
plt.imshow(ex_im_pen)
plt.axis("off")
```
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









## Architecture
We used PyTorch to create a model with an LSTM layer which has a dropout, in addition to a fully connected layer.

The model was written in a general fashion: we set all of his layers size, aswell as the mentioned `bptt` and `batch_size` as variables and proceeded to use [Optuna](https://github.com/optuna/optuna) to optimize over the validation loss.

This loss, aswell as the training loss, was defined as the MSE between the LSTM's prediction of a minute and the closing price of the next minute, which is of course what we're trying to predict.



## Bot
After training the model and feeding the test data, we use a simple bot that tries to buy the stock before it surges and sell it before it gets down. That way, it theoretically can have a profit despite the fact that the stock eventually gets to a lower price than its starting price.

Every minute the bot checks the percentage of different between the actual price of the current minute and the prediction of of the next minute's price. If the percentage is greater than some predefined positive threshold it generally buys a stock, and if the percentage is lower than some predefined negative threshold it sells a stock. For more specifics we refer to `stock_bot.py` .




## Files in the repository

|File name         | Purpose |
|----------------------|------|
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

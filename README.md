# <center>Bitcoin Price Prediction Using Transformers</center>

In this project we used Transformers architecture - encoder-decoder, to predict Bitcoin value into a chosen future horizon. 
Our raw data holds almost 1 year of Bitcoin prices per minute (closing, opening, etc.). We extracted more statistics out of the data using common financial technical indicators - Finta, while making sure they are low correlated between them. We then fed the model with the data and trained it to predict the chosen future horizon based on past values.
We optimized the hyperparameters of the model using Optuna.

Model's prediction on the test set:
![alt text](https://github.com/baruch1192/Bitcoin-Price-Prediction-Using-Transformers/blob/master/images/Test_Prediction.png)
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

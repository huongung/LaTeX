# Chapter 2
DL is representation-learning method with multiple level of representations, obtained by composing nonlinear modules that transform representation at one level to another level. These layers are not designed by human, leant from data

AI: set of algorithms to solve the problems that human can perform intuitively, but hard for the computers

ML tends to be specifically in pattern recognition and learnt from data

The word 'neural' is the adjective form of 'neuron', and 'network' denotes a graph-like structure, thereby ANN is computation system that attempts to mimic the neural connections in our nervous system. From graph theory, we know that a directed graph consists of a set of vertices/nodes and a set of edges/ connections that link together pairs of nodes. Each node performs a simple computation. Each connection then carries a signal (the output of the computation) from one node to another which are weighted to indicate the extent to which the signal is amplified or diminished

Given an image, we supply the pixel intensity values as inputs to the CNN. A series of hidden layers will extract the features from input image. The hidden layers are built on hierarchical manner. At first, only edge-like regions are detected in lower level layers. These edge-like regions are used to define corners and contours. Combine corners and contours can lead to abstract objects in the next layers

Traditional feature engineering and ML: `Input --> Feature Engineering --> ML algorithms --> Output`

Performance (Amount of data): Train (8/9) - Val (2/1) - Test (1)

Parameterization: the process of defining necessary parameters of a given model
1. `Data = Data points + these labels`
1. `Scoring function: f(Data  points) -> These labels`
1. Loss function: degree of matching between predictions and ground truth, matching increase `-->` loss decrease, training is to minimize the loss function
1. Weight and bias
- Advantage: Once training done, discard the input and just keep the weight and biases

## Optimization methods and Regularization
The gradient descent methods is an iterative optimization that operates on loss landscape. The GD is popular is because although this optimization algorithms may not be guaranteed to arrive at global minimum, but is often finds a very low value of the loss fn

Bias trick
\begin{equation}
\[\begin{bmatrix}
1&0&0\\
0&1&0\\
0&0&1\\
\end{bmatrix}\]
\end{equation}

$\sum_{\forall i}{x_i^{2}} $

w_1  w_2  w_3     x      x_1       +      b_1                     w_1   w_2   w_3   b_1     x     x_1
w_4  w_5  w_6            x_2              b_2            -->      w_4   w_5   w_6   b_2           x_2
                         x_3                                                                      x_3
                                                                                                   1

Momentum helps weight update to include momentum term thereby obtaining lower loss in less epochs and increase the strength of updates. V_t = gamma V_t-1 + alpha dev_W_f(W)  ,  W = W - V_t
NAG: In momentum, you add up momentum term and move quite fast. As a result, at the bottom, you likely to hit at full speed. NAG is viewed as corrective update to the momentum. V_t = gamma V_t-1 + alpha dev_W_f(W - V_t-1)

Regularization: lessen effects of overfitting and ensure generalize
- L1/L2 regularization used by updating loss fn which adds more params to constrain capacity of a model
- Dropout
- Data augmentation and early stopping
Generalizability is to make correct classifications on data points that they were not originally trained on
L = 1/N sum_(L_i) + lambda R(W) = 1/N sum_(L_i) + lambda sum(sum(W_2_i,j))

NN Basics
Sigmoid: make the weight hard to update when getting gradient 2 tails, since the delta of gradient will be extremely small
ReLU: extremely computationally efficient, but cannot take derivative of the region when value < 0
LeakyReLU: f(x) = x if x > 0 else alpha x
ELU: f(x) = x if x > 0 else alpha(e^x - 1)
Parametric LeakReLU (PReLU): alpha in LeakyReLU can be leant

4 ingredients of NN:
- Dataset: supervised / unsupervised
- Loss function: categorical crossentropy / binary crossentropy
- Model/ Architecture
- Optimization: SGD --> set proper lr and reg strength --> set # of epochs --> momentum/NAG

CNN Basics
CNN may learn to:
- detect edges from the raw pixel data in the first layers
- use these edges to detect corners and contours (shapes/blobs) in middle layer
- use these corners and contours to detect higher level features such as facial structures, parts of car in the highest layers of the network
- last layer will use these high level features to predict regarding the contents of image
Benefits:
local invariance: classify image which contains an object regardless of where it is in the image thanks to pooling module which identifies regions of input with high response to a particular filters
compositionality: model composes layers that builds on prev layers, stack these building blocks to create CNN
local connectivity: layers will only be connected to a small region of the layer unlike FC layers so save huge amount of parameters

Batch Normalization: normalize the activations of a given input before passing it into the next layer. At the test time, mean and variance batch are superseded by its running averages to avoid biased from the final mini-batch. BN reduces the # of epochs to train and stabilize training but slow down the time to train

Dropout: randomly disconnect with the given prob the connection between 2 FC layers. After 1 pass of for-and-backward, dropout is resampled. Dropout can reduce overfitting because it explicitly altering architecture at training time to ensure that no single node in the network is responsible for activating when given pattern presents and there are multiple redundant nodes activated when a given pattern presents and increase the generalizability

VGGNet is uniquely using 3x3 throughout entire architecture, the use of small kernels help VGG generalize classification problems outside what networks was originally trained on

LR Scheduler
Simply find an area of the loss landscape with reasonably low loss is good enough. If keeping lr high, it's likely to overshoot these areas of low loss, so decrease lr enable network to descend into areas of the loss landscape that are more optimal
Default LR Scheduler of Keras: INIT_LR / (1 + INIT_LR / EPOCHS x Current epoch x Steps_per_epoch)
LR_(epoch + 1) = INIT_LR x Factor ** (epoch + 1 / drop_every)
LR(epoch + 1) = INIT_LR x (1 - (epoch / MAX_EPOCH))**POWER

Over/underfit
Underfit: model cannot obtain enough low loss on train and fails to learn underlying patterns
Overfit: model learn too well and fails to generalize validation data
To prevent the overfit:
- Reduce the complexity/capacity of the model
- Apply reg or LR Scheduler

Regularization is any modification we make to learning algorithms that is intended to increase its generalization. Regularization can be divided into:
- Modify the network architecture - dropout
- Augment data passed into the network(data augmentation)
Data augmentation: generate more data by modifying versions of input to learn more robust features by Translation, Rotation, Changes in scales, Shearing, Horizontal flip

There are 2 types of transfer learning:
- Treat networks as arbitrary feature extractors. We stop forward propagation at certain layer, extract the values from the network at this time, then use them as feature vector
- Remove the FC layers, place a new FC on top of CNN and fine-tune weights

HDF5: Data is firstly defined in groups, where a group can hold datasets. Dataset can be multi-dim array of homogeneous data type (integer, float, str). HDF5 allows to access and slice rows from multi-tb datasets

Flower-17: has only 80 images/class but in general we need to have 1k-5k images/class
Flower-17: fine-grained classification task b/c all categories are very similar; in other words, they share a significant amount of common structure. Fine-grained classification tends to be most challenging, so models need to learn extremely discriminating features

For e.g., Input image of 224 x 224 x 3 after going through VGG -> 1000, but after going thru feature extraction in transfer learning --> 7 x 7 x 512 = 25088, then train off-the-shelf ML model (Linear SVM, Logistic) to obtain the classifier

Advanced Optimized Methods
SGD modifies all params in a network equally in proportion to a given lr, but some values in each updates are much larger than the other ones which may lead to the unbalance in the final weights, and might result in the overfitting in prediction. 

1. Adagrad
Adagrad adapts the lr to the network params. Larger updates are performed on params that change infrequently while smaller updates are done on params that change frequently
cache = cache + (d_W)^2
W -= lr x d_W / (sqrt(cache) + eps)
By examining cache, we can oversee which params are updated frequently/infrequently. However, cache is also the weakness of Adagrad because at each mini batch, the squared gradients are accumulated at denominator. Then, this accumulation keep growing and when dividing small number (gradient) with very large number (cache) will result in infinitesimally small update such that network can't learn any pattern, so that's why Adagrad is rarely seen

2. RMSProp
RMSProp tries to lessen the negative effects of a globally accumulated cache by converting cache into an exponentially weighted moving average
cache = decay_rate * cache + (1 - decay_rate) * (d_W)^2
W -= lr x d_W / (sqrt(cache) + eps)

EWNA: is the way to weight the param in cache. In particular, the most current parameters are weighted most in the cache, then the weights are decreasing gradually for less current parameters
v_1 = beta x v_0 + (1 - beta) x theta_1
v_2 = beta x v_1 + (1 - beta) x theta_2 = beta x (beta x v_0 + (1 - beta) x theta_1) + (1 - beta) x theta_2

Adam: an extension to RMSprop with momentum added to it
m = beta_1 x m + (1 - beta_1) x d_W
v = beta_2 x v + (1 - beta_2)  x d_W^2
W = W - lr x m /(sqrt(v) + eps)
Update identical to RMSProp, but using smoothed version of gradient rather than the raw gradient d_W. The reason why SGD is selected although it converges slower than other advanced optimizations just because people are more familiar with SGD. Moreover, the rate of convergence doesn't matter when compared to performance of the model

Fine-tune networks
Cut-off the head of CNN and replace them by a new set of FC. From there, all layers below the head are frozen so their weights can't be updated. Then train the network using a very small lr. Then, we may unfreeze the rest of the network and continue training. So, fine-tuning allows us to apply pre-trained networks to recognize classes that they were not originally trained on. If new FC layers are not warmed up by freezing the below head, the rich and discriminating features of these layers may be destroyed. But the downside of fine-tuning is to choose the FC layers

model.layers --> layer.name  -> layer.weights --> weight.name --> weight.shape
base_model --> head_model --> model --> freeze layer --> compile --> train --> predict --> unfreeze --> compile --> train

Improve Accuracy with Ensembles
Ensemble methods is process of taking multiple classifiers and aggregate them into one big meta-classifier. By averaging multiple ML model, one big meta-classifier can outperform a single model chosen at random. It may be the case that 1 individual model chosen at random has a better performance than the big classifier, but since there is no criterion to select this model

Advanced Optimization Methods
Strategy to split data: For instance, our model tries to classify cat images from mobile app, but the # of image is only 10K images, while the cat image we crawl online is 200K image. We should build val and test set has 2.5K and 2.5K images, and 205K image for training
                        Similar dataset                     Different dataset

Small dataset     Feature Extraction + classifier     Feature extraction using lower level + classifier

Large dataset               Fine-tuning                     Train from scratch

Not only is HDF5 capable of storing massive datasets, but it's also optimized for I/O ops, esp for extracting batches/slices

Every config file should have
1. Paths to the input folder
2. Total # of class labels
3. Info on the training, val and test sets
4. Paths to the HDF5 datasets
5. Paths to output model, plots, logs

FE by ResNet followed by using Logistic Regression to classify
AlexNet only obtain 94%, not enough to break into top-25 leaderboard, let alone the top-5. To obtain it, we need to to apply transfer learning via feature extraction (ResNet 50) trained on ImageNet

GoogLeNet
Introduced in Going deeper with Convolutions. It's important because the model capacity is tiny compared to VGG thanks to removing FC layers and using Global Average Pooling. Moreover, this paper utilize network in network or micro-architecture when constructing the overall macro-architecture  where output from one layer can split into can split into a # of various paths and rejoined later
This paper contributes the Inception module which enables CNN to learn multiple filter size
Inception module (and its variants)
General ideas: it's hard to decide the size of the filter. Should they be 5 x 5 or 3 x 3 or learn local feature 1 x 1. Instead, why not learn them all and let the model decide. In Inception module, we learn all three 5, 3, 1 and concat the resulting feature maps along the channel dim. This process enables the GoogLeNet to learn both local features via smaller convolution and abstract feature -> multi-level feature extractor
The first branch in the Inception Module learns local features. 
2nd branch first applies 1 x 1 to learn local features and reduce dim, then larger convolution 3 x 3 to reduce computation amount by dim reduction
E.g. 28 x 28 x 192   -----> (5 x 5, same, 32 filters) -----> 28 x 28 x 32, so we will have 32 filters with dims of 5 x 5 x 192, and slide 28 x 28 times on the input, so we have (28 x 28) x 32 x (5 x 5 x 192)
E.g. 28 x 28 x 192   -----> (1 x 1, same, 16 filters) -----> 28 x 28 x 16 -----> (5 x 5, same 32 filters) ---> 28 x 28 x 32, so the amount of computation: (28 x 28) x 16 x (1 x 1 x 192) + (28 x 28) x 32 x (5 x 5 x 16)
3rd branch is same with previous branch, but this larger convolution is 5 x 5
The last branch performs 3 x 3 max pooling with stride = 1. Although Max Pooling can be replaced by Conv layer for reducing the spatial dim and obtain higher accuracy. But, in this case, Pool is added because it's thought that they were needed. The output of the Pool is fed to 1 x 1 convolution to learn local features
Finally, 4 branches were concated along the channel
So, why lr = 5e-3, and 70 epochs with linearly reducing lr is selected. Based on previous training, 80 - 100 is too long so it easily gets overfitting. After training on lr = 1e-3, we may not train hard, but swap to 1e-2 then overfitting ---> train with 5e-3

Tiny ImageNet Challenge
After ex1, switch SGD optimizer to Adam because wasn't convinced that network needed to be deeper with default lr of 1e-3
After ex2, try to deepen network with fourth Inception Module with adding reg=2e-4

ResNet
ResNet uses residual module to train CNN to depths previously impossible. For e.g., VGG16 and VGG19 were considered very deep. However, ResNet can train > 100 layers network. To do this, ResNet uses smarter initialization (Xavier/ He) along with identity mapping
Residual Module
ResNet relies on a micro architecture called Residual Module. Pooling layers are used extremely sparingly, instead convolution with strides > 1 are used to not only learns, but reduce output volume spatial dim. ResNet rather uses average pooling followed by FC module
Going deeper: Residual Modules and Bottlenecks
The original residual module relies on identity mappings which take the original input of the modules and add it to the output of a series of ops
Original Residual Module: X ---> 3 x 3, 64 --->(relu) 3 x 3, 64 ---> Identity map ---> ReLU->
                             |--------------(64-d)------------------^
Bottleneck Residual Module: 
X ---> 1 x 1, 64 --->(relu) 3 x 3, 64 ---> (relu) 1 x 1, 256 ---> Identity map --> (relu) -->
    |----------------------(256-d)----------------------------------^
This type of residual learning framework enables us to train deeper than previous network, and can learn faster with larger LR --> Common to see initial LR = 1e-1
Bottlenecks are an extension to the original versions. We have 3 cons rather than 2 convs. The first and last conv are 1 x 1, plus # of filters in first 2 convs = 0.25 # of filters in the last conv
Why it's called bottleneck? If input of the residual module is M x N x 128. Notice that 64 < 128 means that we are reducing the volume size via 1 x 1 and 3 x 3. This result benefit of leaving 3 x 3 with smaller input and output dim. However, in the last layer, we increase the dim again
Re-thinking the Residual Module
Residual Module with bottleneck is implemented by Conv -> ReLU -> BN
Pre-activation residual module: BN -> ReLU -> Conv
L2 Weight decay  (regularization) is extremely important to ResNet because it's depth

When starting a new set of experiments with either a network unfamiliar, a dataset never worked with before. Ctr C method is used. To do this method, I start training with an initial lr, monitor training, and adjust the LR based on the plot.

Fundamentals of Object Detection
Popular object detection methods utilized 3 crucial components:
1. Sliding windows
2. Image pyramids
3. Non-maxima suppression
Object detection pipeline:
1. A sliding window: slides from left-to-right and top-to-bottom
2. Image pyramid: sequentially reduces the size of our input image. Our sliding window runs on each scale of the image pyramid
--> Using bold sliding window and image pyramid will enable our model to report multiple detections for the same object. When obtaining multiple detections surrounding the same object, we need to apply NMS to keep onl the most confident prediction
3. Apply batch processing to the ROIs to ensure object detector runs as fast as possible
Specifically,
1. Sliding window: take the window region (ROI) and pass it thru our CNN to determine if the window contains an object of interest to us (step and windows_size)
2. Image Pyramid: multi-scale representation of an image, allows us to find out objects at different scales. So the image is subsampled until the min size being reached (scale ratio between 1.1-1.5)
3. Batch processing: CNNs are most efficient when processing batches
4. NMS: solve overlapping bounding box. The reason we get this problem due to sliding windows. When sliding window starts to get closer to an object, it starts to report a high(er) probability for this object. When combined with image pyramids, we will have bounding boxes at multiple scales
NMS works by computing overlap between bounding boxes, then suppressing bounding boxes which don't have highest probability and large overlapping area with it
--> When combining image pyramid and sliding window, we can find objects at both varing locations and varying scales
Downsides to treating a NN trained for classification as an object detector:
- Sliding windows + image pyramids are incredibly slow
- Tedious to tune scale for image pyramid and step size for sliding window
- Due to this tediousness, we can easily miss objects in pur images
--> Rather than, we can train en end-to-end deep learning object detectors

Deep Dream: Running a pretrained CNN in reverse
1. We start with an input image
2. Process the input image at different scales, called octaves
3. For each octave, we maximize the activation of an entire layer sets, mixing the results to obtain trippy effects
Specifically,
- Rather than modifying the weights, we freeze the weights and modify the actual input image.
- When the CNN runs in reverse, the output effects are amplified on the image
- Because lower-level layers generate edge-like regions, intermediate layers for basic shapes and components of objects (eyes, noses), highest-level layer for complete interpretation: dogs, cats --> the highest-level layers of the network is selected
Procedure: original image --> define loss --> calculate gradient --> define function to calculate gradient and loss given original image
Procedure: Given applying gradient ascent on current octave image. To avoid blurry images, we apply detail reinjection. This process is the difference (1) the shrunk image (from smaller octave than current octave) that has been upscaled to the current octave and (2) the original image that has been downscaled to the current octave. Then add the result of this subtraction to our image that has already applied gradient ascent
Procedure:
- Firstly, applying gradient ascent on the current octave image, which calculate the loss as well as gradients to apply this gradient on the original image
- Then, implement the detail reinjection which takes the shrunk image (smaller octave than current octave) then upscaled and original image to downscale

Neural Style Transfer
It enables us to guide the input image to visualize features of a content image and also mix the its features with the style of second image
Algorithms: We don't need special layers in the network, don't need to update the weights.Instead, consider the core component of NN: loss function. Therefore, the question isn't what NN do we use, but rather what loss fn do we use?. The answer is 3 component of loss fn: Content loss, style loss, and total variation loss. Each component will be computed individually, then combined in a meta loss. By minimizing the meta loss, we in turn jointly minimize 3 losses.
Content loss: Since the higher level layers of the network captures abstract objects, a good starting point of content loss is to examine the activations of these high level layer. So we need to
1. Utilize a pre-trained network (based on Imagenet)
2. Select a higher level layer of the network to serve as our content loss
3. Compute the activation of this particular layer for both the content and style image
4. Take the L2-norm between these activations
Since the higher level layer captures the abstract objects, this loss function ensures that generated output will at least look similar to the content image
Style loss: While content loss used only a single layer, our style loss use multiple layers to construct a multi-scale representation of the style and texture. Obtaining this multi-scale representation allows us to capture the style at lower level layer, mid-level layer, and high level layer. In particular, we will compute the correlations between the activations of layers via Gram matrix. A gram matrix is the inner product of style image, thereby forcing the style of the output image to correlate with the style of style image
Total-variation loss: operates solely on the output image which more aesthetically pleasing style transfers by encouraging spatial smoothness across the output image
It is weighted combinations of all 3 losses: loss = alpha * D(style(orig), style(gen)) + beta * D(content(orig), content(gen)) + gamma * tv(gen)
First component is our style loss. We compute the distance D between the style feature of style image and output
Second component is content loss which computes the distance between content feature representations of content image and output
Overal Procesdure: 3 original image --> define loss --> calculate gradient --> define function to calculate gradient and loss given original image --> Use one optimzation methods to optimize one of 3 original image based on the loss and gradients

GAN
These network can be used to generate synthetic images that are near identical to their ground truth
In order to generate synthetic images, we use 2 NN during training
1. A Generator that accepts an input vector of randomly noise and produces an output imitation image that lok similar to an authentic image
2. A discriminator which tries to determine if a given image is an authentic or fake
Therefore, training both of these networks at the same time
So, we can take the example for GAN:
Context: Jack is counterfeit printer (generator) and Jason is an employee of US treasury (discriminator)
1. Jack print fake bills and then mix both, then show them to Jason
2. Jason classify the bills as fake or authentic, give the feedback to Jack how he can improve this counterfeit printing
General training procedure: Most GAN are trained for 6 stpes
1. Randomly generate a vector
2. Pass this noise through the generator to generate actual image
3. Sample authentic images from our training set and then mix them with synthetic images
4. Train our discriminator using mixed set to label them as real or fake
5. Again generate random noise, but we purposely label each noise as real
6. Train the adversarial model using the noise vectors and 'real' images even they are not real
The reason it works because:
1. We have frozen the weights of the discriminator at this stage 6, so the discriminator is not learning when updating the weights of the generator
2. We try to fool the adversarial model into being unable to determine which images are real or synthetic, so the feedback from the discriminator will allow the generator to learn how to produce an authentic images
Guideline and best practices when training GAN:
1. GANs are notoriously hard to train due to an evolving loss landscape. At each iterations, we:
1. Generate random images and then train the discriminator to label correctly
2. Generate additional synthetic images, but this time tries to fool adversarial model, so based on the feedback, we update the weight of generator
So we have  2 losses to minimize
1. The discriminator
2. The adversarial model
Since the loss landscape of generator in adversarial model can be changed due to the discriminator, we end up with dynamic system, So, instead of seeking a min loss value, we try to find the equilibrium between two losses.
Following architecture guidelines for more stable GAN:
1. Replace any pooling layers with strided convolutions
2. Use BN in both the generator and discriminator
3. Remove FC layer in deeper networks
4. Use ReLU in the generator except for the final layer which tanhs is used
5. Use LeakyReLU in the discriminator
Additional guidelines:
1. Sample random vectors from normal rather than uniform
2. Add dropout to the discriminator
3. Add noise to the class labels when training the discriminator
4. To reduce the artifacts in the output, we use kernel size divisible by the stride
5. In case adversarial loss rises while discriminator one fails to zero, try to reduce lr of discriminator and increase its dropout

Super Resolution CNN (SRCNN)
The most significant attributes are listed:
1. SRCNN are fully convolutional, so we can put any image size
2. We trained for filters, not accuracy to enable us to upscale an image --> final accuracy is inconsequential
3. They don't require solving an optimization. After a SRCNN has learnt a set of filters, it can apply forward pass to obtain the output. We don't have to optimize a loss function on a par-image basis to obtain output
4. They are totally end-to-end
Because SRCNN is to learn a set of filters that allow us to map low resolution to higher output, we need 2 sets of images patches:
1. A low resolution patch that will be the input to the network
2. A higher resolution patch that will be target for the network
In practice, the process of SRCNN are:
1. First need to build dataset of low and high resolution input patches
2. Train a network to learn to map the low resolution patches to their high resolution counterparts
3. Create a script to loop over each input images and pass them through the network to predict the output

PERFORMANCE gains using multiple GPU(s)
Training performance is heavily dependent on the PICe bus on your system. In general, training with 2 GPUs tends to improve ~1.8x. When using 4 GPUs, performance scaled to 2.5-3.5x. Architectures that are bound by computation (larger batch size increasing with # of GPUs) will scale better with multiple GPUs as opposed to networks that rely on communication(i.e., smaller batch size). Training speed won't scale linearly with # of GPUs

WHAT IS IMAGENET?
Imagenet is actually a project aimed at labelling can categorizing into 22k categories based on a defined set of words and phrases. At that time, there are 14M images. Each word/ phrase inside WordNet is called symnonym set (or synset) which have 1000+ images per synset

ILSVRC
The goal of the image classification in this challenge is to train a model that can classify an image into 1000 separate object categories, some of which are considered fine-grained classification. Models are trained on ~1.2M training images with another 50K images for validation. While Pascal VOC limit 'dog' to only a single category, ImageNet instead includes 120 different breed of dogs, so required to be discriminative enough to determine what species of dog.
ImageNet DevKit
The DevKit contains:
- An overview and stats for the dataset
- Metadata for the categories(build image filename to class label mappings)
- MATLAB routines for evaluation

PREPARING the IMAGENET dataset
Inside ILSVRC folder, we have 3 directories: Annotation, Data, and Image Sets. `Annotation` is used for localization challenge (i.e., object detection). `Data` directory is more important, we find the sub-directory `CLS-LOC` and 3 more sub-folders: `train`, `val` and `test`
The `test` directory contains 100K images for testing
```bash
ls -l | head -n 10
```
The labels for the testing set are kept private
The `train` directory is organized according to synset ID which maps to a particular object such as `gold fish`, `bald eagle`. For e.g., the wordnet ID n044... consists of 1300 images of 'tench'
```bash
ls -l .../train/n044.../*.JPEG | wc -l
```
The `val` directory
No class label accompany with the `val` image. But, we have `val.txt` which provides us with the mapping from filename to class label

RULE for TRAINING on MXNET
Step 0: Construct the config file which include all data path as well as important parameters
Step 1: Analyze the dataset to build up the text file `.lst` which include `index`, `label of image` and `path to image` for separately training, val and test set
Step 2: Use the script which uses `.lst` files to make them into `.rec` file which MXNET model uses
Step 3: Set the logging config, as well as any additional file: `means` RGB, batch_size based on # of GPUs
Step 4: Construct training and validation image iterator through `mx.io.ImageRecordIter`
Step 5: Construct new model or load it from restart
Step 6: Define the batch as well as epoch end callback, and metric
Step 7: Fit the model by providing it train and val iter, metric, callbacks, defined optimizer (rescale_grad=1/batch_size to avoid mean after batch GD), defined initializer, arg_params, aux_params, begin epoch, and # of epoch

Due to lack of plotting methods, we need to define the method to plot based on the log. The log has format like:
```log
INFO:root:Start training with [gpu(0)]
INFO:root:Epoch[0] Batch [500]	Speed: 880.59 samples/sec	accuracy=0.005148	top_k_accuracy_5=0.020453	cross-entropy=7.656780
INFO:root:Epoch[0] Batch [4500]	Speed: 883.96 samples/sec	accuracy=0.052617	top_k_accuracy_5=0.149516	cross-entropy=6.030526
INFO:root:Epoch[0] Resetting Data Iterator
INFO:root:Epoch[0] Time cost=1397.293
INFO:root:Saved checkpoint to "checkpoints\alexnet-0001.params"
INFO:root:Epoch[0] Validation-accuracy=0.089616
INFO:root:Epoch[0] Validation-top_k_accuracy_5=0.228092
INFO:root:Epoch[0] Validation-cross-entropy=5.336269
INFO:root:Epoch[1] Batch [500]	Speed: 884.90 samples/sec	accuracy=0.068352	top_k_accuracy_5=0.182094	cross-entropy=5.763559
```
`re.findall(r'Epoch\[(\d+)\]', log_content`: 
- Match the character `Epoch`
- `\[`: match the character `[`
- 1st Capturing Group `(\d+)`
    - `\d` matches a digit (equal to `[0-9]`)
    - `+` matches between one and unlimited times
- `\]`: match the character `]`

For every epoch extracted from above, We iterate:
- `Epoch\[1\].*accuracy=([0]*\.?[\d]+)`: for accuracy of epoch 1:
    + `Epoch\[1\]` match the words: `Epoch[1]`
    + `.*` in which `.` matches any character, and `*` matches between zero and unlimited
    + `accuracy` matches the word: `accuracy`
    +   1st Capturing Group ([0]*\.?[\d]+)
        - `[0]*`: match a digit 0 with # from zero to unlimited
        - `\.?`: match the character '.' with # from zero to one
        - `[\d]+`: match the digit from 0 to 9 with # from one to unlimited times

To evaluate the model:
Step 1: provide necessary file, e.g., json file, or path to checkpoint
Step 2: Construct testing iteration based on `mx.io.ImageRecordIter`
Step 3: Load the model from checkpoint and bind them by `model.bind()`
Step 4: Define the metric and score the input data

Network training rule: Allow network to train, monitor the progress, approx 10 epochs. In most case, we need the context of 10-15 epochs before making the decision that a network is indeed overfitting or undefitting. By lowering the lr, we are allowing the network to descend into lower areas of loss because previously lr was too large for the optimizer to find these regions. Keep in mind that the goal of training a network is not necessarily to find a global minimum, rather to find a region where a loss is significantly low
We are not evaluating experiments on the testing data. Only do this when confident  we've obtained a high performing model
AlexNet: 
- Experiment #1: Place BN layers before the activation and standard ReLU rather than ELUs to obtain a baseline for model performance SGD(lr=1e-2, momentum=0.9, wd=5e-4)
- Experiment #2: PLace BN layers after the activation and kept using ReLU (same optimizer config)
- Experiment #3: Replace ReLU for ELU, add 1-2% increase

TRAINING VGGNET on IMAGENET
BY using only 3x3 convolutional layers stacked on top of each other in increasing depth. Reducing the spatial dim is accomplished through the usage of MaxPooling. 2 FC layers each with 4096 nodes (and dropout between) are followed by a softmax classifier
VGG is often used for transfer learning as the network demonstrates ability to generalize the dataset it was not trained on
But training VGG from scratch is a pain because it's brutally slow to train and the weight are large (>500MB)
Implement VGGNet
When implementing VGG, the authors tried vaiants of VGG that increased in depth. There are 6 variants from 11 weight layers to 19 weight layers
VGGNet Config: https://neurohive.io/wp-content/uploads/2018/11/Capture-564x570.jpg
We notice 2 patterns:
- Only use 3x3 filters
- As the depth of the network increases, the # of filter learned double each time max pooling is applied to reduce volume size. The notion of doubling # of filters each time while decreasing spatial dims is of historical importance in deep learning literature. The reason we double # of filters is to ensure no single layer block is more biased than the others.If we reduce the spatial dim without increasing # of depth, our layers become unbalanced and potentially biased because layer earlier may influence our output classification more than layer deeper
Due to its challenge in depth, the networks were simply too deep for basic random initialization. Therefore, the authors came up with `pre-training` to train deeper variants
Pre-training is actually training smaller versions of network architecture with fewer weight layers first, then using these converged network weights as the initialization for the larger, deeper networks. In case of VGG, the authors first trained for config A, 11 weight layers (quite low loss, but not SOTA), then the weights from VGG11 were then used as initialization to config B, VGG13. The excluded layers are randomly initialized while the remainder were simply copied from the pre-trained VGG11. This pattern continue to config D, VGG16, and config E, VGG19. But this technique required more time to apply, although it's still clever trick because it requires to tuning and training hyper-params upto N separate networks
But now, we can rely on a good initialization function. Instead of pure random weight initialization,we now use Xavier Glorot or He/MSRA, so we can skip this pre-training phase
For VGGNet, batch size is reduced from 128 in AlexNet to 32due to VGG16 depth, so can't pass as many image batches through the network at one time
When training VGG16, it was critical that We considered the experiments run by other researhers. Through these work, I was able to avoid running additional, expensive experiments, and applied:
1. Skip pre-training in favor of better initialization methods
2. Use MSRA/ He initialization rather than standard ReLU
3. Use PReLU activation functions rather than Glorot/ Xavier
The biggest downside (besides how long it takes to train) is resulting model size (>500MB). For resource constrained device, this 500MB model size can be a huge burden. In these types of situations, we prefer very small model sizes

TRAIN GoogLeNet on IMAGENET
Both GoogLeNet and VGGNet were the top performers in ILSVRC 2014; however, GoogLeNet has added benefits of being significantly smaller than VGG16 and 19 with only 28.12MB. VGGNet has the own advantage of better generalization and practically achieving high classification. It's also worth mentioning that many researchers have had trouble replicating the original results obtained by Szegedy
The Inception module is a 4 branch micro-architecture. The purpose of Inception module is to learn multiscale features (1x1, 3x3, 5x5 filters) then let the network decide which weights are the most important based on the optimization algorithm. The 4th branch in the Inception Module is called th epool projection which applies 3x3 max-pooling followed by 1x1 convolution. The reasoning behind this branch is that the SOTA circa 2014 applied heavy usage of MaxPooling. But we now know that MaxPooling is not a requirement in network,we can instead reduce volume size through Conv layers; however, this was prevailing at this time. The output of 4 branches are concat along the channel dim, then passed into the next layer
For output size of last Inception Module which is 7 x 7 x 1024, we can use Global Average Pooling to downside to 1 x 1 x 1024.
Small amount of weight decay (wd=2e-4) as recommended by Szegedy and initialize by Xavier/ Glorot method
Note: Although GoogLeNet just slightly beat out VGGNet, many researchers have found it hard to repreduce these exact same results
Experiment #1:
Firstly, We used recommended config by the authors. We used SGD optimizer with an initial lr of 1e-2, as well as moemntum term of 0.9 and a L2 weight decay of 2e-4
Experiment #2:
Given the extreme volatility of 1e-2, We should restart training completely on base lr of 1e-3 to help smooth. I used exact same network architecture, momentum, and L2 regularization as prev experiment. This approach led to steady learning, but painfully slow
Experiment #3:
We dicided to swap out the SGD optimizer for Adam with base lr of 1e-3. Although we are unaware of required to obtain the VGG-level accuracy, we could continue to explore:
1. Swap out ReLUs for ELUs
2. Use MSRA/ He initialization in combination with PReLUs

TRAIN ResNet on IMAGENET
ResNet introduced the concept of residual modules and identity mapping which allowed us to train networks > 200 layers on ImageNet
Understanding ResNet
The cornerstone is residual module consisting of 2 branches. The first is a shortcut which connects the input to an addition of the 2nd branch. It was found that bottleneck residual modules perform better, esp when training deeper network. Moreover, they found that by applying pre-activation, higher accuracy could be obtained. The first CONV consists of 1 x 1 filters, the 2nd of 3 x 3 filters, and the 3rd CONV of 1 x 1 filters. Furthermore, the # of filterslearned by first 2 CONVs is 1/4 # learned by the final CONV. In this chapter, we will be using the bottleneck + pre-activation of the residual module[https://res.cloudinary.com/dqagyeboj/image/upload/v1578709320/Screenshot_from_2020-01-11_09-21-31_rkcydr.png]. Particularly, ResNet50 is the one having 50 weight layers: 2 + (3 x 3) + (4 x 3) + (6 x 3) + (3 x 3)[https://cdn-images-1.medium.com/freeze/max/1000/1*I2557MCaFdNUm4q9TfvOpw.png?q=20]. After last convolutinal layer, average 7 x 7 pooling is then applied (to remove the FC) followed a softmax classifier. It's interesting to note that no dropout applied in ResNet. The reason we perform the CONV on `act1` rather than `shortcut` because the `data` will be batch normalized by BatchNorm as well as Activation before doing shortcut. Based on the table, our `stages` = [3, 4, 6, 3] and `filters` = [64, 256, 512, 1024, 2048]
Experiment #1:
We start with an inital lr of 1e-1, then lower by an order of magnitude when overfitting, momentum term of 0.9, and L2 weight decay of 1e-4 (as recommended by the authors). Since we are training a very deep neural network, we should use He initializer
Future experiments should consider being more aggressive with regularization, including weight decay, data augmentation, and even apply dropout

TRAIN SQUEEZENET on IMAGENET
We decreaes model size by applying a novel usage of 1 x 1 and 3 x 3 convolutions, along withno FC layers. Moreover, the model size can be reduced to 0.5MB by model compression, also called weight pruning and sparsifying a model (setting 50% of the smallest weight value across layers to zero).
Fire Module is critical microachitecture responsible for reducing model params and amintaining a high level of accuracy
Fire Module
The fire module relies on an expand and reduce phase consisting only 1 x 1 and 3 x 3 convolutions
The squeeze phase of the Fire Module learn a set of 1 x 1 filters -> ReLU. The # of squeeze filters < volume size input to the the squeeze; therefore, can be considered as dim reduction. Moreover, by using 1 x 1 filters, we can learn local features and the spatial relationship pixels has amongst its channels. Typically, we would use a larger 3 x 3  or even 5 x 5 kernel to learn features capturing spatial info of pixels lying close together in the input volume. Finally, after reducing dim (since # of filter < input depth size), we add more nonlinearity as the ReLU is applied after 1 x 1 convolution
After that, the output is fed into expand. During that stage, we learn a combination of 1 x 1 and 3 x 3 convolutions. The 3 x 3 convolution allowed us to capture spatial dim from the original 1 x 1 filters. Generally, We learn N, 1 x 1 expand filtersand N, 3 x 3 expand filter (N = 4x than # of squeeze filters). The output of 1 x 1 and 3 x 3 expands are concat across the filter dim. The 3 x 3 conv are zero-padding of 1 to ensure unchanged dim
[https://res.cloudinary.com/dqagyeboj/image/upload/v1578726837/Screenshot_from_2020-01-11_14-03-10_risboj.png]
Training SqueezeNet
The authors recommend to apply lr of 4e-2 but I found this lr to be too large. It easily leads to extremely volatile in learning, hard to converge, so I used base lr of 1e-2, momentum of 0.9 and wd of 2e-4
Experiment #1:
Instead of ELU, I used ReLU with lr of 4e-2 (as recommended by author). Accuracy was plummeted all the way down to 0.9% and reconver again
Experiment #2:
Because 4e-2 was so volatile, I decided to reduce lr to 1e-2
Experiment #3:
The prev experiments are examples of failed experiments that sound good in our head, but when actually apply it, it fails miserably. Given that micro-architecture (GoogLeNet, ResNet) benefited from having BN, I updated the SqueezeNet to include BN after every ReLU. But, eventually. BN don't help to increase performance
Experiment #4:
I decided to swap out to ELU. ALthough we can replicate author's acc, this success hinged on replacing 1e-2. After searching, other DL researchers also report 4e-2 large. Experiments like this show how challenging to replicate the authors' acc. Depending on DL library and versions of CUDA, we may get different results even if implemtation is identical to the authors. Some DL libraries implement layers differently, so there maybe underlying params we are unaware of
Note: be sure to log these results in a journal, so we can revisit the results and examine what we can tweak to boost accuracy. In most cases, this will be our lr, regularization, weight init, act function

SUMMARY of TUNING
Always start with ReLU (or recommended by authors) to obtain baseline. Then tweak other hyper-params to the network, including lr scheduling, regularization/ wd, weight init, act function, even BN. Once optimzing network as far as we think we can, swap out ReLU to ELU. One exception, if training a very deep CNN and use He Init, we should consider swapped to PReLU

CASEE STUDY: EMOTION RECOG
Based on the raw dataset, we construct the HDF5 to feed data to Keras model
Implement VGG like network
1. The Conv layer in the network will only 3 x 3
2. We'll double the # of the filters learned by each CONV layer deeper we go in the network
To aid in training the network, we apply some priori knowledge gained from experiencing with VGG
1. We should initialize CONV using He Initialization, doing so will enable network to learn faster
2. Since ELU and PReLU have been shown to boost classification accuracy throughout all of experiment, so let's start with an ELU rather than ReLU
In this case, the reason why validation set has its Image Data Generator because `rescale` attribute. We stored these images as raw, unnormalized RGB images, meaning that pixel values existed in range [0, 255]. However, it's common to practice either (1) mean normalization or (2) scale the pixel intensities down to a more constricted amount. Luckily, we can do option (2) by `rescale=1/255` in `ImageDataGenerator`
Experiment #1:
As always do with 1st experiment, We aim to establish a baseline that we can incrementally improve upon. We started with SGD optimizer with base lr of 1e-2, momentum and NAG are applied. During the training, it's interesting to note that as soon as we lowered the lr from 1e-2 to 1e-3, the network effectively stopped learning
Experiment #2:
Given that SGD led to stagnation in learning, We decided to swap out to SGD for Adam, base lr of 1e-3. but sharp divergence -> clear overfitting, but overall result still better than SGD
Experiment #3:
A common cure to overfitting is to gather more training data, so we can applu data augmentation to reduce overfitting. In this experiment, We kept Adam optimizer. But the downside is that we aren't seeing any dramatic gains in accuracy
Experiment #4:
1. We swapped Xavier for He because He init tends to work better for VGG
2. Replace all ReLUs with ELUsto boost accuracy
3. Merge label to ease class inbalance. Again, the Adam optimizer with base lr of 1e-3
Summary: it was important that our CNN be
1. deep enough to obtain high accuracy
2. but not so deep that it would be impossible to run in realtime

Case Study: CORRECTING IMAGE ORIENTATION
The reason we are using transfer learning via feature extraction from pre-trained CNN is 2-fold:
1. To show that the filters learned by CNN trained on ImageNet are not rotation invariant across a full 360 spectrum
2. Transfer learning via feature extraction obtains the highest accuracy when predicting image orientation
It's common misconception that the individual filters learnt by CNN are invariant to rotation. Instead, the CNN is able to learn a set of filters that activate when they see a particular object under a given rotation. CNNs are not fully rotation invariant; otherwise, it'd be impossible for us to determine the orientation of an image strictly from the features extracted during the finetuning phase

Age and Gender classification
Fine-grained classification in which small difference may have different prediction. This thesis presents a CNN-based method to improve accuracy of age and gender classification of face image
Face detection is performed using Haar Cascade classifier. The classifier is trained from a lot of positive and negatives images. The algorithm has 4 stages:
1. Haar feature selection   2. Creating integral images     3. Adaboost training        4. Cascading classifiers
Initially, the algorithm needs a lot of positive images of faces and negative images without faces to train the classifier. The first step is to collect the Haar Features from data. A Haar feature considers adjacent rectangular regions at a specific location in a detection, sums up the pixel intensities in each region and calculates the difference between them. But how do we select the best features? This is accomplished using a concept called Adaboost which both selects the best features and trains the classifiers that use them. This algorithm constructs a strong classifier as a linear combination of weighted simple weak classifiers. The process as follows:
During the detection phase, a window of the target size is moved over the input image, for each sub-image, Haar features are calculated. The difference is then compared to a learned threshold that separates non-faces and faces
Facial landmark
Facial landmark tries to determine fundamental points of the face when given an input image. There are 2 steps to detect facial landmarks. Firstly, it uses face detection methods (Haar cascade classifier) to detect faces. Then, it detects crucial facial points on ROI based facial landmark detector. To train it, an ensemble of regression trees is trained based on training data which facial landmark is manually labelled, (x, y)-coords in the image to estimate the facial landmark positions directly from the pixel intensity
Face alignment
It consists of 2 steps, which are identifying the geometric structure of faces when given input data and then based on image processing techniques (translation, scale, and rotation) it attempts to obtain an alignment of the face. My way to align the face is obtained using computing the difference between angle of eyes region and horizontal line. The detected faces in the dataset is aligned before passing into ML models
During the training time, after reading all images in benchmark dataset, face detector is firstly applied to detect a single face. The face is extracted from input image and normalized by face alignment which rotates and translates extracted face. The face is then randomly cropped to dim of 227 to be eligible fot CNN so that the network never sees the exact image to increase the generalizability of the model
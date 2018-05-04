--------------------
# Feature engineering 
Feature engineering is a craft used to transform real-world, raw data into features that are:
*  Representative of problem’s properties; 
*  Are well      comprehensible to the predictive algorithms used toward modelling that problem. 

#### But what defines a Feature?
Data tabulated as observations, variables taking values are referred as attributes, and select few of which make up to features of learning problem. All attributes aren’t features as features are more pronounced in whole context of a problem. Features form the basis of observing and understanding structure of a problem meant to be modeled. For instance, In visual recognition, images are observations whereas only an edge, contour form a feature. In language processing, words, sentences are observation whereas word count, wordVec forms a feature [1].
The aim is to get the gist of the data one has aggrevated and churn it through a predictive model to obtain the best results. Thus the idea states taking the voluminous, ambiguous formed data and shape it into a reasonable set of variables having some sort of correlation with targeted output.
This calls for simplifying relationship between features and the output to be mapped. As simple correlations between the features and target give the case of easy learning but constrained model accuracy for a particular algorithm, contrarily if the target is a complex function of the features, often as the raw data is usually not in the form that is amenable to learning, yields poor accuracy, which can be improved significantly by drawing more correlations, which are otherwise indirect and abstruse. As the features that look irrelevant in isolation may be relevant in combination [2].

##### It’s a process by itself involving:
[[a]] . Brainstorming features; [[b]] . Deciding what features to create; [[c]] . Creating features; [[d]] . Studying impact of features on model accuracy; [[e]] . Iterating on features [3].

##### Here are few methods adopted in feature engineering:
*	##### Feature extraction:
it is performed on problems involving voluminous, raw data, far too much to be modeled by predictive algorithms directly. Ex- image, textual data. Hence techniques like PCA are employed for dimensionality reduction into its principal components to a relatively lower dimensional space. Ex- converting quantities like mass, distance, timing to new units or decomposition of one into its constituent components.
* ##### Feature selection:
Data involving attributes irrelevant and usually redundant towards modelling a problem thus need to be removed, paving room for features that are more important for model accuracy.
* ##### Feature construction:
Manually creating features requires spending time with sample data pondering the structure of the underlying problem, causality of particular target in regard to available data. In structured data, it often means combining and splitting features to create new ones. Ex- for an image classification task one can create binary features for each color present as 0 or 1.
* ##### Feature learning/representation learning:
DL methods utilizing autoencoders and restricted Boltzmann machines have been shown to learn feature abstraction in an unsupervised or semi-supervised manner, with applications in image classification, speech recognition [1].

Feature engineering is an intellectual task in succession to data cleaning in a ML pipeline, where one need to draw insight into the how the given raw input data maps to target output labels. The Insights could be aided by data visualization tools, finding correlation between conspicuous and latent variables, values that affect the state of target variable.
Learning problems such as regression, multiclass classification problems encompassing structured data rely heavily on feature engineering whereas Methods like deep learning dealing with unstructured image data work fine even with negligent feature engineering as subsequent layers of a deep neural network tend to pick representative features of an input image on its own.

##### Reference:
[1] https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/

[2] https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf

[3] Big Data in Education, Week 3 Feature Engineering; Ryan Baker.
__________
__________
# Feature Maps:
A feature map is a result of convolving a kernel spatially over sub-regions of a previous input layer (be it image or hidden layer) throughout its depth. A filter of definite size traverses constantly along pixels in horizontal and vertical, each movement of which is computed as one single node/neuron activation and the final output of convolution is collectively referred as a feature map.
For instance, a filter of shape 5 × 5 x 3 convolving over an image of 32 × 32 x 3, with a displacement of 1 pixel will yield a feature map of 28 × 28 x 1. Intuitively when this convolution is carried with multiple filters across same input image, initial feature maps would capture simple curves, whereas deeper layers would capture combination of curves forming circles.

![Feature Map](http://deeplearning.net/tutorial/_images/cnn_explained.png)
#####  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Feature map for filter w1 & w2

#### what is a kernel/Filter?
a small spatial structure more like a tensor consisting values initialized randomly across width, height and depth (say 5x5x3), that are subsequently updated gauging patterns like edges, curves, colour gradients in the input data as soon as the algorithm starts learning.
The kernel in CNN is analogous to receptive field of V1 neurons meant for low level abstraction in biological prospect. That is useful while dealing with high-dimensional data as that of images, as it imparts local connectivity to input layer and facilitates reusing parameters for every convolution. 
The convolution neural network consists of output volumes made by multiple filters convolving full depth through input volume, where each filter results into one such activation map capturing a particular feature, and several such activation maps are stacked to perform convolution by next set of filters subsequently.

![Feature Map](http://cs231n.github.io/assets/cnn/stride.jpeg)
##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Feature map with convolution of stride 1 & 2  



##### The size of a feature map is decisive of convolution between kernel and input and following factors:
* Stride: the value by which filter slides spatially across input volume. For instance, stride 1 moves the filters one pixel at a time.
* zero-padding: It involves padding borders of input volume with zeros, this helps preserving or control the spatial size of the feature map.

##### Computing feature map spatial size:
The spatial size of the feature map can be computed as a function of the input volume size (V), filter size (F), the stride (S) and the amount of zero padding (P) on the border.
The number of output neurons is given by
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[(V−F+2P)/S] +1

For Ex- a filter of 3x3 convolves over an image of 9x9 with stride 2 and padding 0 would result into a 4x4 feature map.

##### References:
[1] http://deeplearning.net/tutorial/lenet.html

[2] http://cs231n.github.io/convolutional-networks/ 

------------------------------------------------

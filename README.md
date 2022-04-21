# Practica_Imagen

# Neural style transfer

### Convolutional Neural Network:

Convolutional Neural Networks (CNNs) are a category of Neural Network that have proven very effective in areas such as image recognition and classification. CNNs have been successful in computer vision related problems like identifying faces, objects and traffic signs apart from powering vision in robots and self driving cars.

CNN is shown to be able to well replicate and optimize these key steps in a unified framework and learn hierarchical representations directly from raw images. If we take a convolutional neural network that has already been trained to recognize objects within images then that network will have developed some internal independent representations of the content and style contained within a given image.

In 2014, the winner of the ImageNet challenge was a network created by Visual Geometry Group (VGG) at Oxford University, as a basis for trying to extract content and style representations from images, naming it after them.

The VGG net where shallow layers learns low level features and as we go deeper into the network these convolutional layers are able to represent much larger scale features and thus have a higher-level representation of the image content.

Neural style transfer can be implemented using any pre-trained convnet. Here we will use the VGG19 and VGG16 networks. VGG19 is a simple variant of the VGG16 network, with three more convolutional layers.

Using multiple convolution layers with smaller convolution kernels instead of a larger convolution layer with convolution kernels can reduce parameters on the one hand, and the author believes that it is equivalent to more non-linear mapping, which increases the Fit expression ability.

For the VGG16 network, every step they applied kernals for 2–3 time and then applied a max pooling layer. In our case, we want to account for features across the entire image so we get rid of the maxpool which throws away information and replace those layers for ones that compute the Average Pooling instead. Each time the number of kernals are doubled from the previous layer meaning that each time we trying to extract more and more features. At the end the of the network three fully connected layers are used to limit the relu activation function grow. Dropout is also implemented for reduce overfitting of model.

 ![VGG16](/ImagenesNotebook/vgg16-architecture.png)

Regarding the VGG19 network, it has 16 convolutions with ReLUs between them and five maxpooling layers which we will also substitute for the Average Pooling. The number of filter maps of the convolutions start at 64 and grow until 512. After the convolutions, there is a linear classifier made-up three fully-connected (fc) layers with dropout (SHK * 14) between them, the first two have 4096 features while the last one has 1000. The last fc layer is connected to a softmax which maps each value to the probabilities of belonging to each of the 1000 classes of the ImageNet competition. 

 ![VGG19](/ImagenesNotebook/vgg19-architecture.png)

### Style Transfer

Style Transfer is a technique of modifying one image in style of another image. We are implementing Gatys style transfer which was originally released in 2015 by Gatys et al. The neural style transfer algorithm has undergone many refinements and spawned many variations. Neural style transfer consists in applying the "style" of a reference image to a target image, while conserving the "content" of the target image:

Style refers to the textures, colors, and visual patterns in an image while the "content" is the higher-level macrostructure of the image. 

The key point behind style transfer is same idea that is core to all deep learning algorithms: we define a loss function to specify what we want to achieve, and we minimize this loss. We want to achieve: conserve the "content" of the original image, while adopting the "style" of the reference image. The theoretical loss function would be the following:

We can construct images whose feature maps at a chosen convolution layer match the corresponding feature maps of a given content image. We expect the two images to contain the same content — but not necessarily the same texture and style.

#### Loss

##### The content loss

Given a chosen content layer l, the content loss is defined as the Mean Squared Error between the feature map F of our content image C and the feature map P of our generated image Y.

When this content-loss is minimized, it means that the mixed-image has feature activation in the given layers that are very similar to the activation of the content-image. Depending on which layers we select, this should transfer the contours from the content-image to the mixed-image.

As you already know, *activations from earlier layers in a network contain local information about the image*, while *activations from higher layers contain increasingly global and abstract information*. Therefore we expect the "content" of an image, which is more global and more abstract, to be captured by the representations of a top layer of a convnet.

##### The style loss

Now we want to measure which features in the style-layers activate simultaneously for the style-image, and then copy this activation-pattern to the mixed-image.

One way of doing this, is to calculate the Gram-matrix(a matrix comprising of correlated features) for the tensors output by the style-layers. The Gram-matrix is essentially just a matrix of dot-products for the vectors of the feature activations of a style-layer. This inner product can be understood as representing a map of the correlations between the features of a layer. These feature correlations capture the statistics of the patterns of a particular spatial scale, which empirically corresponds to the appearance of the textures found at this scale. If an entry in the Gram-matrix has a value close to zero then it means the two features in the given layer do not activate simultaneously for the given style-image. And vice versa, if an entry in the Gram-matrix has a large value, then it means the two features do activate simultaneously for the given style-image. We will then try and create a mixed-image that replicates this activation pattern of the style-image.

Hence the style loss aims at preserving similar internal correlations within the activations of different layers, across the style reference image and the generated image. In turn, this guarantees that the textures found at different spatial scales will look similar across the style reference image and the generated image. The loss function for style is quite similar to out content loss, except that we calculate the Mean Squared Error for the Gram-matrices instead of the raw tensor-outputs from the layers.

### TL:DR

In short, being the content image the one we wish to modify and the style reference image the one we obtain the style from, we can use a pre-trained convnet to define a loss that will:

* Preserve content by maintaining similar high-level layer activations between the target content image and the generated image. The convnet should "see" both the target image and the generated image as "containing the same things".
* Preserve style by maintaining similar correlations within activations for both low-level layers and high-level layers. Indeed, feature correlations capture textures: the generated and the style reference image should share the same textures at different spatial scales.

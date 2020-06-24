## Discussion 24 June

1. A combination of 9 different networks. Each network takes in a single view image (in grayscale) of the 3D data and does feature extraction on that view. The full network combines the extracted features of the 9 different networks for classification.

2. In the first approach, the input will be slightly modified. The binary images will be generated from the point cloud data, squeezing/expanding them along x, y, z (time) to make inputs of fixed size and time length and fed into the network. This way loss of useful information can be avoided.

3. An autoencoder trained on psudo generated stalled samples and their non-stall versions. Task of the autoencoder is to learn how to fill pixels to remove the stall, matching with the non-stall sample. In case of stall-less examples, the autoencoder will do nothing. In the testing process, autoencoder takes test samples, and tries to fill up the stall regions. In case there is significant difference between input and output (stall was filled), the original input was stalled. If there is not much difference, then input was not stall

4. Some point cloud based approaches, still looking into those. So far pointNet is the most promising network possible to use. Also tried 3D convolution but results were not satisfactory.


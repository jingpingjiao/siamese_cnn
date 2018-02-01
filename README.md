# Facial Similarity with Siamese Network
Pytorch implementation of Siamese networks that learns to classify its inputs, the neural networks learns to differentiate between two people

## Dataset
The data used is Labeled Faces in the Wild Home (LFW). You can access and download the full dataset [here](http://vis-www.cs.umass.edu/lfw/#views)

Please place the data in the ```PROJECT_FOLDER/lfw/```. The resulting folder structure looks like this: ![img](https://github.com/JiaoJingPing/siamese_cnn/blob/master/imgs/folder.png)

Splitting for training and testing has been done. You can view the split in test.txt and train.txt with rows containing image files paths of two images and then a 1 if they are the same person and 0 if not.

Feel free to split the dataset however you want, but please be reminded to update the train.txt and test.txt.

## Network Architecture
A Siamese Network is a CNN that takes two separate image inputs, and both images go through the same exact CNN. Then we use a sort of loss function to compute the similarity between two output. [Gupta on Hackernoon](https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7) has a nice illustration for the network. You can check out his article for more explanation. 
The detailed architecture can be found in `architecture.md`.

## Loss Function
In this implementation, two losses function are explored, namely Binary Cross-Entropy(BCE) and Contrastive Loss. Depends on the loss function chosen, the network architecture for the output layer is different. 

### Binary-classification with BCE
For BCE, the outputs from two CNNs are concatenated and then put through a sigmoid activation function to return either 1 or 0.

### Contrastive-loss
Instead of passing the output to a sigmoid activation function, we take output from two CNNs to compute a contrastive loss. Again, you can read more about it [here](https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e)

## Image Augmentation
To improve the generalization power of the model, data augmentation is implemented.  This means, during the training loop, as images are read in, with some probability apply a random transform to the images before running it through the network.
The transform can consist of some random combination of: mirror flipping, rotation, translation and scaling.


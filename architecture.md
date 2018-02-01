1. Convolution Layer
Input: image of size 128x128x3 (3 for RGB color channels)
Parameters: 64 features of size 5x5, stride of (1,1), padding=2
Output: tensor of size 128x128x64
2. ReLU (in-place)
3. Batch Normalization Layer
64 features
4. Max pooling layer
Input: tensor of size 128x128x64
Parameters: none, perform 2x2 max-pooling with (2,2) stride
Output: tensor of size 64x64x64
5. Convolution Layer
Input: 64x64x64
Parameters: 128 features of size 5x5, stride of (1,1), padding=2
Output: tensor of size 64x64x128
6. ReLU (in-place)
7. Batch Normalization Layer
128 features
8. Max pooling layer
Input: tensor of size 64x64x128
Parameters: none, perform 2x2 max-pooling with (2,2) stride
Output: tensor of size 32x32x128
9. Convolution Layer
Input: 32x32x128
Parameters: 256 features of size 3x3, stride of (1,1), padding=1
Output: tensor of size 32x32x256
10. ReLU (in-place)
11. Batch Normalization Layer
256 features
12. Max pooling layer
Input: tensor of size 32x32x256
Parameters: none, perform 2x2 max-pooling with (2,2) stride
Output: tensor of size 16x16x256
13. Convolution Layer
Input: 16x16x256
Parameters: 512 features of size 3x3, stride of (1,1), padding=1
Output: tensor of size 16x16x512
14. ReLU (in-place)
15. Batch Normalization Layer
512 features
16. Flatten Layer
Input: tensor of size 16x16x512
Parameters: none, simply flatten the tensor into 1-D
Output: vector of size 16x16x512=131072
Note: this simple layer doesn't exist in pytorch. You'll have to use view(), or implement it yourself.
17. Fully Connected Layer (aka Linear Layer)
Input: vector of size 131072
Parameters: fully-connected layer with 1024 nodes
Output: vector of size 1024
18. ReLU (in-place)
19. Batch Normalization Layer
1024 features

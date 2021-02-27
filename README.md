# Dog-Breed-Identification
The project is about Convolutional Neural Network that performs better than the average human to identify dog breeds. Given an image of a dog, the algorithm will produce an estimate of the dog’s breed. If given an image of a human, the output would be an estimate of the closest-resembling dog breed. Along with exploring pre-trained CNN models such as VGG, Resnet, Inception for classification, the goal is to get an insight how to finetune and use a pretrained network and apply it to a new task using transfer learning.

[Kaggle Dog Breed Indendification](https://www.kaggle.com/c/dog-breed-identification)

## Background
Image Classification is now considered a fairly solved problem thanks to state-of-art deep learning models which were presented thrugh ImageNet competition. ImageNet is a visual Dataset that contains more than 15 million of labeled high-resolution images covering almost 22,000 categories.VGG-16 is a Convolutional Neural Network model that was proposed by the researchers of the University of Oxford. VGG-16 had secured 1st runners up position in the 2014 ImageNet competition with an error rate of 7.3%. In VGG-16 the main characteristic is that it uses several 3×3 kernel-sized filters consecutively. The hidden layers of the network leverage ReLU activation functions. VGG-16 is however very slow to train and the network weights, when saved on disk, occupy a large space.
ResNet (Residual Network) was created by the Microsoft Research team and won the 2015 ImageNet competition with and the error rate of 3.57%.The idea of using numerous hidden layers and extremely deep neural networks was implemented by a lot of models but then it was realized that such models were suffering from vanishing or exploding gradients problem. Inception was Google’s developed image classification deep learning model that was the winner of the 2015 ImageNet challenge with an error rate of 6.67%.The idea behind the architecture was to design a really deep network with 22 layers by choosing 1×1 convolutional along with ReLU to increase computational efficiency by reducing the dimensions and number of operations.

## Step 0: Import Datasets
* 13233 total human images
* 8351 total dog images

## Step1. Detect Humans
Using OpenCV's pre-trained face detector - haar cascade face detector
* red image --> convert BGRto Gray -->Gaussian Blur -->  add counding box for face region
* performance
Human detected as Human : 0.99
Dod detected as human: 0.2
2nd haar cascade detector
human detected as human : 1.0
dog is detected as human : 0.62
Keep the 1st one for human detector
## Step2. Detect Dogs
Using PyTorch's pre-trained VGG-16 Model, output index is 151-268 for dog identification.
Data Transform pipeline
```python
data_transform = transforms.Compose([
                                    transforms.RandomResizedCrop(250),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))])
```
VGG-16 Human is detected as Dog : 0.04
VGG-16 Dog Accuracy : 0.99
Other human detector

## Step3: Build own CNN
batch_size =32, learn_rate=0.01
1. Using same data transform pipleline
No drop out: just 3 (Conv+relu + Max pooling ) + 2 fully connected layer

2. Added Dropout (0.25) for between fc1 before the output

3. Using Data Augmentation (Horizontal Flip, Random Rotation)
```python
data_transform = transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticallFlip(),
                                    transforms.RandomRotation(20),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
```
Epoch: 30 	Training Loss: 3.335449 	Validation Loss: 4.074869
Test Accuracy: 10.52% (88/836)

## Step4: Using Transfer Learning
Pytorch's pre-trained model
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])
```
There are three pre-trained models explorered .
* VGG 16-layer model (configuration “D”) [“Very Deep Convolutional Networks For Large-Scale Image Recognition”](https://arxiv.org/pdf/1409.1556.pdf). VGG16 is constructed with all the conv kernels are of size 3x3 and maxpool kernels are of size 2x2 with a stride of two. VGG's reduced number of trainable variables helped faster learning and more robust to over-fitting.
 ```python
torchvision.models.vgg16(pretrained=True) ## VGG16
```
* ResNet-50 model from [“Deep Residual Learning for Image Recognition”](https://arxiv.org/pdf/1512.03385.pdf) ResNet architecture makes use of shortcut connections do solve the vanishing gradient problem.
```python
torchvision.models.resnet50(pretrained=True) ###Resnet-50
```
* Inception v3 model architecture from [“Rethinking the Inception Architecture for Computer Vision"](http://arxiv.org/abs/1512.00567). Each inception module consists of four operations in parallel ; 1x1 conv laer, 3x3 conv layer , 55 conv layer, max pooling layer. if the images in the data-set are rich in global features without too many low-level features, then the trained Inception network will have very small weights corresponding to the 3x3 conv kernel as compared to the 5x5 conv kernel.

```python
torchvision.models.inception_v3(pretrained=True) ###Inception v3 model
```
0. Pytorch's pretrained model
```python
torchvision.models.vgg16(pretrained=True) ## VGG16
torchvision.models.resnet50(pretrained=True) ###Resnet-50
torchvision.models.inception_v3(pretrained=True) ###Inception v3 model
```
> As for Inception v3 model, Important: In contrast to the other models the inception_v3 expects tensors with a size of N x 3 x 299 x 299, so ensure your images are sized accordingly.
> RGB images of shape (3 x H x W), where H and W are expected to be at least 299
```python
preprocess = transforms.Compose([
transforms.Resize(299),
transforms.CenterCrop(299),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

1. Resnet50 from Pytorch
CrossEntry Liss, SGD optimizer , learn rate = 0.005
```python
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.fc.parameters(), lr=0.005)
```
Epoch: 10 	Training Loss: 2.405513 	Validation Loss: 1.714103 ....
Epoch: 20     Training Loss: 1.907578     Validation Loss: 1.498481
Validation loss decreased (1.595233 --> 1.498481).  Saving model ...
Test Loss: 1.679269
Test Accuracy: 71% (601/836)

## Discussion 
By trasfer learning using ResNet50 pretrained model, Performance is increased by 71%. That shows it is great way to use pre-trained model especially for image classification problem thanks to ImageNet's massive data set and augmented lables. Resnet takes long time to train however by suing Cuda, it helped to train efficiently on GPU environment. When training loss is dramatically lower than validation loss, it would be better to increase Dropout rate since this shows overfitting stage. When training loss is very similar to validation loss, underfitting case is suspicious. In that case, adding more layers would help. I explored number of epochs along with learning rate  until delta training loss is saturated. 
#### Next Task 
[x] 1. Object Classification
[x] 2. Image Classification and Localization
[x] 3. Object Detection
[x] 4. Semantic and Instance Segmentation
[x] 5. Object Tracking
[x] 6. Image Captioning 
[x] 7. Image Generation
[ ] 8. Image Enhancement/Colorization
[x] 9. Image Style Transfer
[ ] 10. Image Construction

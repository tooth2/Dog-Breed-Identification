# Dog-Breed-Recognizer
Using various kinds of deep learning methods such as CNN layers from scratch vs. using Pre-trained model such as VGG, ResNet, Inception ,this project is to identify dog breed when it comes to dog identification whereas as human , to recommend most similar dog breed to a human face in the image.

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
in_transform = transforms.Compose([
                                    transforms.RandomResizedCrop(250),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))])
```
VGG Human is detected as Dog : 0.05
VGG Dog Accuracy : 0.98
Other human detector

Inception


## Step3: Build own CNN
batch_size =32, learn_rate=0.01
1. Using same data transform pipleline
Performance:
Epoch: 20     Training Loss: 3.915724     Validation Loss: 4.209987
Validation loss decreased (4.257086 --> 4.209987)
No drop out: just 3 (Conv+relu + Max pooling ) + 2 fully connected layer
Test Loss: 4.178142
Test Accuracy:  7% (62/836)

2. Added Dropout (0.25) for between fc1 before the output
Performance:


3. Using Data Augmentation (Horizontal Flip, Random Rotation)
```python
in_transform = transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(20),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
```
Performance:


## Step4: Using Transfer Learning
1. Resnet50 from Pytorch
CrossEntry Liss, SGD optimizer , learn rate = 0.001
Epoch: 20     Training Loss: 1.907578     Validation Loss: 1.498481
Validation loss decreased (1.595233 --> 1.498481).  Saving model ...
Test Loss: 1.422683
Test Accuracy: 70% (587/836)

2. Inception3?

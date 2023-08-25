# Project 4: Group Project - Image Classifiction

## Problem Statement
As waste management leaders, RepublicServices understands the importance of efficiency and accuracy in the recycling process. Our goal is to develop a machine learning solution that aligns with my company's core objectives and can bring a significant improvement to our waste classification process.

Using a comprehensive dataset from Kaggle, with over 22,000 training images and 2,500 validation images, our solution is to design a model to accurately classify items in images as either organic or recyclable. This is pivotal for ensuring that recyclable materials are free from contamination with organic waste, a challenge that many recycling facilities face.

Integrating the model into recycling facilities can lead to a more streamlined waste sorting process. This not only reduces reliance on manual labor but also significantly minimizes human error. The goal is to have a model that achieves over 90% accuracy when classifying "organic" vs. "recycle" images to ensure effective batch sorting.

---

## Dataset
- Over 22,000 training and 2,500 validation images from Kaggle (https://www.kaggle.com/datasets/techsash/waste-classification-data) categorized into ‘Organic’ and “Recyclable’.
- Number of training images: 22,564
  - Organic: 12,565
  - Recyclable: 9,999
- Number of testing images: 2,513
  - Organic: 1,401
  - Recyclable: 1,112
- Total file size: 222.24 MB


## Data Preprocessing
- Converted color of images to RGB format
- Changed size of images to 128 x 128 pixels
- Normalized pixel values to 0-1
- Binarized class labels

  
## EDA
- The distribution of pixel values within each color channel of an image, reveals the predominant color used in an image visible to the human eye
- Color intensity / brightness of pixels for images in Train and Test are similar
- Color intensity / brightness of pixels for images that are labeled ‘Recyclable’ is brighter than that of ‘Organic’ images for both Train and Test
- Red is the most predominant color channel, followed by Green for both ‘Organic’ and ‘Recyclable’ images

---

## Modeling
 - Established a baseline score by examining the target class distribution
 - Conducted multiple iterations of CNN model development with varying number of layers, regularization techniques, epochs, and batch sizes
 - Conducted multiple iterations of CNN model development using a pre-trained model (VGG16) and custom 'top' layers
 - Employed data augmentation techniques to apply various random transformations to the original images, creating new training samples
 - Regularization techniques used:
   - L2 Regularization
   - Drop out
   - Early Stopping
 - Evaluated prediction results using various classification metrics and analyzed misclassified images
 
---

## Conclusion and Improvement Opportunities
- Our goal was to build a model that outperformed the pre-trained model, VGG16. After customizing the 'top' layers, we got an accuracy of approximately 90% for both the training and validation datasets.
- By employing data augmentation, various regularization techniques, and adding more Convolutional and Dense layers, we achieved higher training and validation accuracy (0.9137) compared to the model that relies on VGG16.
- Several of the misclassified images are challenging to categorize, even for humans. This suggests that the model is performing at or near an optimal level.

### Improvement opportunities:
- There is potential for improvement if we were not constrained by Colab's limitations, such as limited compute units, memory, and storage.
- We could explore other pre-trained models, such as ResNet50, Inception, EfficientNet, etc.
- Delve deeper into the misclassified images to discover trends and patterns. For instance, utilize interpretation techniques like LIME (Local Interpretable Model-Agnostic Explanations) to gain a better understanding of model predictions.

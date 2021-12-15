# ASLTranslation
CSE 490 Deep Learning Final Project by Manan Gandhi and Neil Kagalwala

# Presentation Video

# Abstract
American Sign Language (ASL) is the universally adopted standard for communication between humans when one is unable to speak. However, less than 1% of the US population is fluent in ASL. Even more so, most deaf and hard of hearing people don't know sign language. One of the biggest challenges is the lack of dynamic and immediate translation between ASL and English. Using the concepts learned in class, we wanted to design a solution for this that would help eliminate the translation barrier between the two languages.  
Our motivation for learning about ASL, specifically its translation, comes from our interest in exploring the intersection of AI and accessibility. With this project, we aim to create a proof of concept for a translation tool that takes in video of sign language and translates it into English text. 

# Problem Statement
We are trying to solve this problem by a complete solution that can be divided into two pieces. First, we will implement a standard multi-class classification algorithm to identify different ASL letters when given an image of the hand-sign. Second, we will implement object-detection methods to process videos of ASL actions that can be passed as images into our classification algorithm.

# Related Work
Initially, we used a pretrained model that had been trained on the COCO dataset, as a baseline for our model for our ASL classification.

https://cocodataset.org/#home

For the main model portion of the project, we used this American Sign Language Dataset. In short, this dataset contains 1728 images of different participants' hands performing different ASL letters. The dataset also contains bounding boxes for specifically the hand in the image that is denoting in the image, which we had hoped to incorporate into our object detection part of the project.

https://public.roboflow.com/object-detection/american-sign-language-letters 

Lastly, we generated self-collected videos to see how effective our solution would be compared to others. Our video took hand-signs of different participants and passed it into our model, to extract a text translation of ASL. 

Other notes
- We were inspired by this hand gesture recognition project done at UT Austin: https://towardsdatascience.com/american-sign-language-hand-gesture-recognition-f1c4468fb177.
While we didn’t end up using their approach, we were specifically interested in their rationale behind the solution and their ideas in approaching both static and dynamic movement in ASL. 
- Finding appropriate datasets for our goal was one of the biggest challenges that we faced. We initially wanted to use videos of ASL as our dataset, and use this dataset to fine-tune our classification model: http://vlm1.uta.edu/~athitsos/asl_lexicon/. After many hours trying to clean up the video data available from this source from Boston University and UT Arlington, we were unable to properly clean and format the data enough, and we also realized that we had a partial dataset of not enough iterations of different sequences of letters to properly affect our model. 

# Methodology
Our approach was to use the feature layers from ResNet18 pre trained on COCO, then build out our own fully connected layers and train those. We found that this gave better results than training the entire model after pretraining, and since there were less weights to change, the speed was improved as well. 

To handle the data, we resize each image to be 200x200, then use random affines and color jitters to help combat overfitting. This was necessary because earlier approaches were achieving 98% training accuracy but only 60% test accuracy. This reduced the training speed, but we noticed a smaller gap between train and test accuracy. 

For the final step, we used a video processing library called Katna to extract key frames from videos of sign language. We then pass these key frames through the model, and generate a prediction for which letter is being displayed, with some recorded degree of confidence. The key frame analysis can be shaky, so we only pick out layers that have a confidence level of above 50%.

# Experiments and Evaluation
We ran 8 experiments, with different network configurations (frozen layers, data augmentations, hyperparameters) to find which ones worked best. 

Once we found the best model, we fine tuned our parameters (learning rate, momentum, batch size, etc.) to improve our model even more. Then, we measured the effectiveness of our model by calculating the training/test accuracy using a predefined split in our initial data set. 

Lastly, we tested our model on new ASL videos. We compared the generated translation from our solution to the expected translation computed manually by us, and determined qualitative accuracy which we go into further below.

# Results
We managed to achieve 98% training accuracy and 60% test accuracy on our initial dataset. Though we expected these results to translate well to other videos, we have so far achieved limited success. 

TODO: Graphs: Train Accuracy, Test Accuracy, Loss

When we passed in images extracted from videos into our model, we faced multiple challenges. While we initially planned to create a customer SSD for object detection, we faced challenges in implementing it to our needs and with the time constraints. So we decided to use key frame extraction as another way to extract images from the video. However, this was not exactly picking the best images out. To combat this, we also tried to extract more images that anticipated from the videos. For example, if we knew our video had ~20 or so ASL characters, we would extract 100 images. We then used conjoined confidence intervals to make better predictions of the 20 expected characters. Our dependency on using an external library to do this was mainly the issue here, as we believed we could have achieved much better results had we extracted more exact images. 

# Demo


# Next Steps
While our results were not as expected, we hope to explore and continue working on our solution to make it more expected. Here are a couple extensions of this project that we hope to complete and visit again in the near future:
1. Object Detection using SSD: SSD would have allowed us to combine object detection and classification with high accuracies in real time. We attempted to implement this, and spent a good amount of time on it. However, we were unable to implement a custom SSD that would work on our dataset after significant effort. Therefore, we decided to take a different approach given the time constraints of this project. 
2. Incorporate LSTM and RNN: We make predictions about letters based on the image of a hand. However, this ignores the context of previous images detected. A next step could be to combine image classification with something like a GRU and hidden layer to incorporate previous results in the current classification.
3. Better Keyframe Analysis. Keyframe analysis methods work differently for different genres of media. For exa	mple, a keyframe analysis method for movies would likely omit scenes at the beginning in favor of scenes at the end, and prefer scenes with high amounts of movement and action. In contrast, we would want key frame analysis for sign language processing to have an even distribution of images over time, heavily preferring images where movement has slowed down. Building this custom keyframe analysis could be a major step in improving this project.

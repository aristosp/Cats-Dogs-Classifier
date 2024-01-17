# Cats & Dogs Classifier

A project based on Chapter Four of [Modern Computer Vision with Pytorch]. Create a Convolutional Neural Network in Pytorch classifying images of dogs and cats from [Kaggle].
Example Training Images:

![cat 1](https://github.com/aristosp/Cats-Dogs-Classifier/assets/62808962/7442d848-ab19-4ccf-9254-0fc0467388ff) ![dog 4](https://github.com/aristosp/Cats-Dogs-Classifier/assets/62808962/60451cac-5fe7-49b6-ae70-b456b70021b0)

The training dataset contains 4000 images of each pet, while the test dataset contains 1000 images of each. Images vary in size, so they are all resized to 224 x 224 and are used as RGB. The training set was augmented as well, with the following transformations:
![augments](https://github.com/aristosp/Cats-Dogs-Classifier/assets/62808962/097702d9-cbec-4d02-912b-5a99938a47f1)

The repository contains:
* A utils.py file containing functions regarding the dataset, the model, the training process and various other utilities.
* The main.py file containing the hyperparameters, the loading of the dataset etc.

Example predictions from the test set can be seen in the following image:
![Predictions](https://github.com/aristosp/Cats-Dogs-Classifier/assets/62808962/47dc6a8c-c3e6-4c4f-8451-f0e2b6d3756e)




[Modern Computer Vision with Pytorch]: https://www.oreilly.com/library/view/modern-computer-vision/9781839213472/
[Kaggle]: https://www.kaggle.com/datasets/tongpython/cat-and-dog

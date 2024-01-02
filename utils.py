import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2
import numpy as np
from glob import glob
from random import shuffle, seed


seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class DogCats(Dataset):
    """
    Custom Dataset class, reading training files for each class
    With __getitem__ if transforms are enabled, images are augmented
    """

    def __init__(self, folder, transform=None):
        cats = glob(folder + '/cats/*.jpg')
        dogs = glob(folder + '/dogs/*.jpg')
        self.filepaths = cats + dogs
        shuffle(self.filepaths)
        self.targets = [fpath.split('/')[-1].startswith('dog') for fpath in self.filepaths]
        self.transform = transform

    def __getitem__(self, index):
        image = self.filepaths[index]
        target = self.targets[index]
        im = (cv2.imread(image)[:, :, ::-1])
        im = cv2.resize(im, (224, 224))
        if self.transform:
            im = self.transform(im)
        # Permute dimensions as pytorch expects them to be C X H X W
        # Return normalized image and its label
        return (torch.tensor(im / 255).permute(2, 0, 1).to(device).float(),
                torch.tensor([target]).float().to(device))

    def __len__(self):
        return len(self.filepaths)


def conv_layer(ni, no, kernel_size, stride=1):
    """
    Wrapper function containing the following sequence of layers:
    Conv2D -> ReLU -> Batch Normalization -> 2D MaxPool
    :param ni: number of input channels
    :param no: number of output channels
    :param kernel_size: kernel size
    :param stride: stride
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(ni, no, kernel_size, stride),
        nn.ReLU(),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(2))


def get_model():
    """
    Simple wrapper function creating the model.
    :return: network: the model object
    """
    network = nn.Sequential(
        conv_layer(3, 64, 3),
        conv_layer(64, 128, 3, stride=2),
        nn.Dropout(0.25),
        conv_layer(128, 256, 3),
        conv_layer(256, 512, 3),
        nn.Dropout(0.25),
        conv_layer(512, 512, 3),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(512, 64),
        nn.Linear(64, 1),
        nn.Sigmoid()).to(device)
    return network


def train_per_epoch(train_dataloader, model, optimizer, loss_fun):
    """
    Wrapper function containing the training process for one epoch
    :param train_dataloader: the dataloader for training files
    :param model: the model to be used
    :param optimizer: user defined optimizer
    :param loss_fun: loss function, defined by user
    :return:
    train_accuracy: the accuracy of training samples for one epoch
    training_loss: the loss computed by the loss function, for the training samples during one epoch
    """
    model.train()
    training_loss = []
    per_batch_acc = []
    for _, samples in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training',
                           unit='Batch', position=0, leave=True):
        # Unpack inputs and labels
        x, y = samples[0].to(device), samples[1].to(device)
        # Zero gradients for each batch
        optimizer.zero_grad()
        # Predictions for this batch
        y_predictions = model(x)
        # Compute accuracy
        acc = accuracy(x, y, model)
        # Append per batch accuracy
        per_batch_acc.extend(acc)
        # Compute loss
        loss = loss_fun(y_predictions, y)
        # Add loss to total loss sum
        training_loss.append(loss.item())
        # Back-propagate loss
        loss.backward()
        # Change weights
        optimizer.step()
    # Calculate mean loss and accuracy
    training_loss = np.mean(training_loss)
    train_accuracy = np.mean(per_batch_acc)
    return train_accuracy, training_loss


@torch.no_grad()
def accuracy(x, y, model):
    """
    Simple function returning the accuracy between actual and predicted labels
    :param x: input samples
    :param y: actual class labels
    :param model: model to be used
    :return:
    is_correct: list containing true or false values, where true corresponds to the sample
    being in the dog class and false not being in the dog class, i.e. cat
    """
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    return is_correct.cpu().numpy().tolist()


def prediction_mode(dl, model, desc, loss_function):
    """
    Function containing the evaluation process of the model.
    :param dl: Dataloader object
    :param model: Model to be used
    :param desc: Whether the function is used during validation or testing
    :param loss_function: user defined loss function
    :return:
    avg_v_loss: loss during evaluation for one epoch
    avg_v_acc: accuracy during evaluation for one epoch
    """
    model.eval()
    with torch.no_grad():
        validation_loss = []
        per_batch_val_acc = []
        # Iterate through validation data
        for _, vdata in tqdm(enumerate(dl), total=len(dl), desc=desc,
                             unit='Batch', position=0, leave=True):
            v_input, v_target = vdata
            v_pred = model(v_input)
            # Compute val accuracy
            v_acc = accuracy(v_input, v_target, model)
            per_batch_val_acc.extend(v_acc)
            # Compute val loss
            v_loss = loss_function(v_pred, v_target).item()
            validation_loss.append(v_loss)
        # Compute the average of each metric for one epoch
        avg_v_loss = np.mean(validation_loss)
        avg_vacc = np.mean(per_batch_val_acc)
    return avg_v_loss, avg_vacc


def early_stopping(model, filename, mode):
    """
    Function implementing early stopping techniques, using the mode variable.
    :param model: model to save
    :param filename: path and name of the file
    :param mode: whether to save the model or restore the best model from a path
    :return: NULL
    """
    if mode == 'save':
        torch.save(model.state_dict(), filename)
    elif mode == 'restore':
        model.load_state_dict(torch.load(filename))
    else:
        print("Not valid mode")


def int2txt(label_int, mode):
    """
    Function converting integer labels to text labels
    :param label_int: list containing the labels in integer format
    :param mode: whether predictions or original labels are used
    :return:
    txt: list containing the labels in string format
    """
    txt = []
    if mode == 'initial':
        for int_label in label_int:
            if int_label == 1:
                txt.append('Dog')
            else:
                txt.append('Cat')
    else:
        for pred_label in label_int:
            if pred_label:
                txt.append('Dog')
            else:
                txt.append('Cat')
    return txt

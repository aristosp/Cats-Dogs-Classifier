# V6: Early stopping implemented
# V5: Added two image augmentations, horizontal flip and random rotation
# V4: Random sample predictions added + dropout layer to model + code optimizations
# V3: Create validation function
# V2: Test set is used
# V1: Train Test Split Done
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.data import random_split
from utils import *
from torchsummary import summary
from random import seed, sample
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch import optim
seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Augmentations to implement
transforms = [  # v2.RandomResizedCrop(size=(224, 224)),
    # v2.ColorJitter(brightness=.5, hue=.3),
    # v2.RandomPerspective(distortion_scale=0.6, p=1.0),
    v2.RandomHorizontalFlip(p=1.0),
    v2.RandomRotation(45)]

generator1 = torch.Generator().manual_seed(0)
# Dataset Directories
train_dir = 'training_set/'
test_dir = 'test_set/'
# Dataset objects containing original and augmented files
data = DogCats(train_dir)
data_aug1 = DogCats(train_dir, transform=transforms[0])
data_aug2 = DogCats(train_dir, transform=transforms[1])
# Split train dataset into training and validation
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
batch = 32
train_ds, val_ds = random_split(data, [train_size, val_size], generator=generator1)
# Create augmentations
train_ds_t1, _ = random_split(data_aug1, [train_size, val_size], generator=generator1)
train_ds_t2, _ = random_split(data_aug2, [train_size, val_size], generator=generator1)
# Concatenate the three training datasets into one
train_ds = torch.utils.data.ConcatDataset([train_ds, train_ds_t1, train_ds_t2])
train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=batch, shuffle=True, drop_last=True)
# Lists to save metrics for plots
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
# Define hyperparameters
model = get_model()
loss_fn = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=1e-3)
summary(model, input_size=(3, 224, 224))
epochs = 15
# Initialize variables to use for early stopping purposes
min_loss = torch.tensor(float('inf'))
early_stop = 2

for epoch in tqdm(range(epochs), desc='Epoch', unit='Epoch', position=0, leave=True):
    model.train(True)
    train_acc, train_loss = train_per_epoch(train_dl, model, opt, loss_fn)
    val_loss, val_acc = prediction_mode(val_dl, model, 'Validation', loss_fn)
    # Append to lists for later visualizations
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print('\n')
    print("Epoch {} Training Accuracy {:.2f} %, "
          "Validation Accuracy {:.2f} %".format(epoch + 1, train_acc * 100, val_acc * 100))
    print("Epoch {} Training loss {:.3f}, Validation Loss {:.3f}".format(epoch + 1, train_loss, val_loss))
    # Early stopping conditions
    if val_loss < min_loss:
        min_loss = val_loss
        best_epoch = epoch
        early_stopping(model, "best_model.pth", 'save')
    elif epoch - best_epoch > early_stop:
        print("Early stopped training at epoch %d" % epoch)
        early_stopping(model, "best_model.pth", 'restore')
        break  # terminate the training loop

# Plot Accuracy
epochs = np.arange(epoch + 1)
plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
plt.legend()
plt.grid('off')
plt.show()

# Test model on new unseen data
test_files = DogCats(test_dir)
test_dl = DataLoader(test_files, batch_size=batch, shuffle=False, drop_last=True)
test_loss, test_acc = prediction_mode(test_dl, model, 'Test', loss_fn)

print("Test Accuracy: {:.2f} % , Test Loss: {:.3f}".format(test_acc * 100, test_loss))


# Get a batch of images to predict
images, label = next(iter(test_dl))
txt_labels = int2txt(label, mode='initial')
predictions = model(images)
predictions = (predictions > 0.5) == label
txt_pred_labels = int2txt(predictions, mode='predictions')

# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
indexes = sample(range(0, 31), 12)
for i in range(len(indexes)):
    idx = indexes[i]
    # Convert image to int, with pixels in range of 0-255
    img = images[idx, :, :, :] * 255
    img = img.permute(1, 2, 0).cpu().numpy()
    img = img.astype(int)
    ax = figure.add_subplot(2, 6, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(img)
    # Set the title for each image
    ax.set_title("{} ({})".format(txt_pred_labels[idx], txt_labels[idx]),
                 color=("green" if txt_pred_labels[idx] == txt_labels[idx] else "red"))
# If the prediction has been made correctly, image title is green, else is red
# Title format : predicted label (true label)
plt.show()

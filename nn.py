import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
device = 'cpu'

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
import time
LOSS = []
start = time.perf_counter()
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1)

        scores = model(data)
        loss = criterion(scores, targets)
        LOSS.append(loss)
        print(f"EPOCH [{epoch} / {num_epochs}], loss: {loss}")

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
end = time.perf_counter()
duration = end-start
print(f"done in {duration:0.3} seconds")

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)










images, labels = next(iter(test_loader))
IMG = images
# Convertir en format approprié pour la visualisation
images = images.numpy()
labels = labels.numpy()



# Générer des prédictions simulées (pour la démonstration)
model.eval()
IMG = IMG.reshape(IMG.shape[0], -1)
predictions = model(IMG)

#predictions = np.random.randint(0, 10, 18)

# Fonctions de visualisation comme dans le code fourni
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
  
    plt.imshow(img.squeeze(), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
  
    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100*np.max(predictions_array),
                                         true_label),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
 
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Visualisation des images et des graphiques de prédictions
num_rows = 6
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
num_images_to_display = min(len(images), len(labels), 10)  # Assurez-vous de ne pas dépasser la taille des tableaux

# Ajustement de la boucle pour afficher les images
for i in range(num_images_to_display):
    plt.subplot(2, num_images_to_display, 2*i + 1)
    plot_image(i, predictions.detach().numpy(), labels, images)  # Assurez-vous que np.random.rand(10) est remplacé par les vraies prédictions si disponibles

    plt.subplot(2, num_images_to_display, 2*i + 2)
    plot_value_array(i, predictions.detach().numpy(), labels)  # De même, utilisez les vraies prédictions si disponibles
plt.tight_layout()
plt.show()




loss_values = [loss.item() for loss in LOSS]


epochs = range(1, len(loss_values) + 1)

# Créer le graphique
plt.plot(epochs, loss_values, marker='o', color='b', label='Loss')

# Ajouter des titres et des étiquettes
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Afficher le graphique
plt.show()
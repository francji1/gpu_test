import os
import zipfile
import requests
import warnings
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torchvision import models, datasets, transforms
from torchviz import make_dot
from torchsummary import summary
from torch.utils.data import DataLoader, random_split

from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

import subprocess


def delete_small_files(directory, size_threshold=2048):
    """
    Delete files smaller than a specified size in a directory.

    :param directory: Path to the directory to scan for small files.
    :param size_threshold: File size threshold in bytes (default is 1024 bytes for 1KB).
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if file size is smaller than the threshold
            if os.path.getsize(file_path) < size_threshold:
                print(f"Deleting {file_path}...")
                os.remove(file_path)

delete_small_files('PetImages')


class RobustImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            except Exception as e:
                print(f"Error with image {path}: {e}, skipping.")
                index = (index + 1) % len(self.samples)


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




def create_basic_model(device, transform):
    # Loading a pretrained model (ResNet18) and modifying it for binary classification
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Modifying for 2 classes: Cats and Dogs

    # Moving the model to the specified device
    model = model.to(device)
    print(f"Model moved to {device}")

    # Determining input size based on the transform
    resize_transform = [t for t in transform.transforms if isinstance(t, transforms.Resize)]
    input_size = resize_transform[0].size if resize_transform else (224, 224)

    # Convert the size to the format expected by torchsummary and displaying the summary
    input_shape = (3, input_size[0], input_size[1])
    summary(model, input_shape)

    return model

class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        # Load a pre-trained ResNet
        original_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Everything except the last layer
        self.features = nn.Sequential(*list(original_model.children())[:-2])

        # Flatten operation
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # Final layer for binary classification

    def forward(self, x):
        # Apply the feature extractor
        x = self.features(x)

        # Flatten the output
        x = self.flatten(x)

        # Apply the fully connected layers with ReLU activations
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)

        return x

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.parameters():
            param.requires_grad = True


def create_modified_model(device, transform):
    modified_model = ModifiedResNet()

    # Moving the model to the specified device
    modified_model = modified_model.to(device)
    print(f"Modified model moved to {device}")

    # Determining input size based on the transform
    resize_transform = [t for t in transform.transforms if isinstance(t, transforms.Resize)]
    input_size = resize_transform[0].size if resize_transform else (224, 224)

    # Convert the size to the format expected by torchsummary and displaying the summary
    input_shape = (3, input_size[0], input_size[1])
    summary(modified_model, input_shape, device=device.type)

    return modified_model

def plot_metrics(n_epochs, train_accuracies, test_accuracies, train_precisions, test_precisions, train_recalls, test_recalls, filename):
    epochs = range(1, n_epochs + 1)
    plt.figure(figsize=(12, 4))

    # Check if the lengths of metric lists match n_epochs
    if len(train_accuracies) == n_epochs and len(test_accuracies) == n_epochs:
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_accuracies, 'bo-', label='Train Accuracy')
        plt.plot(epochs, test_accuracies, 'ro-', label='Test Accuracy')
        plt.title('Training and Testing Accuracy')
        plt.legend()

    if len(train_precisions) == n_epochs and len(test_precisions) == n_epochs:
        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_precisions, 'bo-', label='Train Precision')
        plt.plot(epochs, test_precisions, 'ro-', label='Test Precision')
        plt.title('Training and Testing Precision')
        plt.legend()

    if len(train_recalls) == n_epochs and len(test_recalls) == n_epochs:
        plt.subplot(1, 3, 3)
        plt.plot(epochs, train_recalls, 'bo-', label='Train Recall')
        plt.plot(epochs, test_recalls, 'ro-', label='Test Recall')
        plt.title('Training and Testing Recall')
        plt.legend()

    plt.savefig(filename)
    plt.show()



def main():
    # URL of the dataset
    url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'

    # Send a GET request to the URL
    response = requests.get(url)

    # Open the file in binary write mode and save the content
    with open("kagglecatsanddogs_5340.zip", "wb") as file:
        file.write(response.content)

    print("Download completed.")


    # Path to the downloaded zip file
    zip_file_path = "kagglecatsanddogs_5340.zip"

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(".")

    # Checking the contents of the extracted directory
    if os.path.exists('PetImages'):
        print("Extraction successful. Contents:", os.listdir('PetImages'))
    else:
        print("Extraction failed or 'PetImages' directory not found.")


    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset using RobustImageFolder
    full_dataset = RobustImageFolder(root='PetImages', transform=transform)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(full_dataset))  # 80% of the dataset for training
    test_size = len(full_dataset) - train_size  # the rest for testing
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create the basic model
    model = create_basic_model(device, transform)
    model = model.to(device)
    print(model)

    # Create the modified model
    model = create_modified_model(device, transform)
    model = model.to(device)
    print(model)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)


    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=0.001)

    train_accuracies, train_precisions, train_recalls = [], [], []
    test_accuracies, test_precisions, test_recalls = [], [], []
    n_epochs = 5


    def create_requirements_file(file_path='requirements.txt'):
        try:
            # Running 'pip freeze' to get the list of installed packages
            result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE, text=True)

            # Writing the output to the requirements.txt file
            with open(file_path, 'w') as file:
                file.write(result.stdout)

            print(f"'{file_path}' file created successfully.")
        except Exception as e:
            print(f"Error occurred: {e}")

    # Call the function to create the requirements.txt file
    create_requirements_file()


    for epoch in range(n_epochs):
        # Variables to store predictions and labels for the training dataset
        train_labels = []
        train_predictions = []

        # Training loop with progress bar
        model.train()  # Set the model to training mode
        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Training]')
        for inputs, labels in train_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Calculate and print training metrics
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_precision = precision_score(train_labels, train_predictions, average='binary')
        train_recall = recall_score(train_labels, train_predictions, average='binary')
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        print(f'Epoch {epoch+1}/{n_epochs} Training - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}')

        # Variables to store predictions and labels for the testing dataset
        test_labels = []
        test_predictions = []

        # Testing loop with progress bar
        model.eval()  # Set the model to evaluation mode
        test_progress_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Testing]', leave=False)
        with torch.no_grad():
            for inputs, labels in test_progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_predictions.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # Calculate and print testing metrics
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_precision = precision_score(test_labels, test_predictions, average='binary')
        test_recall = recall_score(test_labels, test_predictions, average='binary')
        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        print(f'Epoch {epoch+1}/{n_epochs} Testing - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')

        # After completing all epochs, plot the metrics and save the figure
        plot_metrics(n_epochs, train_accuracies, test_accuracies, train_precisions, test_precisions, train_recalls,
                     test_recalls, 'training_metrics.png')


    # # Plotting the metrics
    # epochs = range(1, n_epochs+1)
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1)
    # plt.plot(epochs, train_accuracies, 'bo-', label='Train Accuracy')
    # plt.plot(epochs, test_accuracies, 'ro-', label='Test Accuracy')
    # plt.title('Training and Testing Accuracy')
    # plt.legend()
    #
    # plt.subplot(1, 3, 2)
    # plt.plot(epochs, train_precisions, 'bo-', label='Train Precision')
    # plt.plot(epochs, test_precisions, 'ro-', label='Test Precision')
    # plt.title('Training and Testing Precision')
    # plt.legend()
    #
    # plt.subplot(1, 3, 3)
    # plt.plot(epochs, train_recalls, 'bo-', label='Train Recall')
    # plt.plot(epochs, test_recalls, 'ro-', label='Test Recall')
    # plt.title('Training and Testing Recall')
    # plt.legend()
    #
    # plt.show()


if __name__ == '__main__':
    main()
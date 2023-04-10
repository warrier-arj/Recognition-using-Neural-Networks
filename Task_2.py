#
"""
Arjun Rajeev Warrier
CS 5330: Pattern Recognition and Computer Vision
Project 5: Recognition using Deep Networks

Task_2.py loads a previously trained network and evaluates.
"""
# import statements
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import cv2


class Net(nn.Module):
    """
    Class to initialize a network model.
    Also contains a forward function to propagate an input through the network
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   # First convolutional layer
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # Second convolutional layer
        self.conv2_drop = nn.Dropout2d()        # dropout layer to set some activations to zero randomly --  robustness
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        A function test a particular input x against the different layers.

        Input: input x
        Output: log_softmax value of final result
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def apply_filters_to_image(image, filters):

    filtered_images = []
    for i in range(filters.shape[0]):
        filter = filters[i, 0].numpy()
        filtered_image = cv2.filter2D(image.reshape(image.shape[1], image.shape[2]), -1, filter, borderType=cv2.BORDER_CONSTANT)
        filtered_images.append(filtered_image)
    return filtered_images


# main function (yes, it needs a comment too)
def main(argv):
    """
    The driver function. Initializes the network and iterates over epochs for training and testing.
    """
    network = Net()

    # Task 2 A
    network.load_state_dict(torch.load('/results/model.pth'))          # Load saved model
    print("Network Loaded from '/results/model.pth' ")

    conv1_weights = network.conv1.weight.data                          # Get the weights of the first layer
    print("Shape of conv1 weights:", conv1_weights.shape)

    fig = plt.figure()                                                 # Visualize the filters using pyplot
    for i in range(conv1_weights.shape[0]):
        filter_weights = conv1_weights[i, 0]
        plt.subplot(3, 4, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(filter_weights)
        plt.title("Filter {}".format(i+1))
        plt.tight_layout()
    plt.show(block=False)

    # Task 2 B
    dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))
    image = dataset[0][0].numpy()

    # Apply filters to the image using OpenCV filter2D function

    filtered_images = apply_filters_to_image(image, conv1_weights)
    filtered_images = apply_filters_to_image(image, conv1_weights)
    fig3 = plt.figure()
    for i in range(len(filtered_images)):
        # Plot the filtered images alongside the filters
        plt.subplot(3, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(filtered_images[i], cmap='gray')
        plt.title("Filter {} ({:.2f})".format(i + 1, conv1_weights[i, 0].sum().item()))

    plt.show(block=False)

    fig2 = plt.figure()
    i = 0
    j = 0
    while j < 2*len(filtered_images):
        filter_weights = conv1_weights[i, 0]
        plt.subplot(5, 4, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(filter_weights)
        plt.subplot(5, 4, j + 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(filtered_images[i], cmap='gray')
        j += 2
        i += 1
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""

End of program

"""
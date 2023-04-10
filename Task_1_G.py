#
"""
Arjun Rajeev Warrier
CS 5330: Pattern Recognition and Computer Vision
Project 5: Recognition using Deep Networks

Task_1_G.py tests a loaded MNIST digit network with custom hand drawn digit inputs.
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


class customTransform:
    """
    Class to transform input images for usage in train and test functions.
    """
    def __init__(self):
        pass

    def __call__(self, x):

        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.resize(x, (28, 28))
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

# main function (yes, it needs a comment too)
def main(argv):
    """
    The driver function. Initializes the network and iterates over epochs for training and testing.
    """
    network = Net()

    # Task 1 G
    # Loading model
    network.load_state_dict(torch.load('/results/model.pth'))
    print("Network Loaded from '/results/model.pth' ")

    # Get dataset for testing
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('/files/MNIST_custom',
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   customTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=10,
        shuffle=True)

    # Testing the model
    network.eval()
    with torch.no_grad():
       for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx == 0:
                output = network(data)
                for i in range(len(target)):
                    print('Outputs for example', i + 1)
                    for j in range(10):
                        print('{:.2f}'.format(torch.exp(output[i][j]).item()), end=' ')
                    print('\nPredicted:', output.data.max(1, keepdim=True)[1][i].item(), 'Actual:',
                          target[i].item())
                fig = plt.figure()
                for i in range(10):
                    plt.subplot(5, 2, i + 1)
                    plt.tight_layout()
                    plt.imshow(data[i][0], cmap='gray', interpolation='none')
                    plt.title("Predicted: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
                    plt.xticks([])
                    plt.yticks([])
                #plt.tight_layout()
                plt.show()
                break



    print("End of Program")
    return

if __name__ == "__main__":
    main(sys.argv)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""

End of program

"""
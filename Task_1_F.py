#
"""
Arjun Rajeev Warrier
CS 5330: Pattern Recognition and Computer Vision
Project 5: Recognition using Deep Networks

Task_1_F.py loads and evaluates a previously trained model on separate testing data and shows predictions.
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


# main function (yes, it needs a comment too)
def main(argv):
    """
    The driver function. Initializes the network and iterates over epochs for training and testing.
    """
    network = Net()

    # Task 1 E
    network.load_state_dict(torch.load('/results/model.pth'))
    print("Network Loaded from '/results/model.pth' ")

    # Getting test data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))])),batch_size=9, shuffle=True)

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
                for i in range(9):
                    plt.subplot(3, 3, i + 1)
                    plt.tight_layout()
                    plt.imshow(data[i][0], cmap='gray', interpolation='none')
                    plt.title("Predicted: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
                    plt.xticks([])
                    plt.yticks([])
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
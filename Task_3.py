#
"""
Arjun Rajeev Warrier
CS 5330: Pattern Recognition and Computer Vision
Project 5: Recognition using Deep Networks

Task_3.py trains the model against the Greek letter dataset with params from the MNIST digit network.
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

# Global Variables used
n_epochs = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 1


def pause():
    # Function whose primary purpose is to pause a program during runtime till any other key is pressed.
    programPause = input("Press <Enter> key to continue...")


class GreekTransform:
    """
    Class to transform input images for usage in train and test functions.

    """
    def __init__(self):
        pass

    def __call__(self, x):
        """

        Function containing transform operations.

        """
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )



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


def train(epoch, network, optimizer, train_loader, train_losses, train_counter):
    """
    A function to train the network.

    Input: epoch, network, optimizer, train_loader, train_losses, train_counter
    Output: No return; modifies training counter and loss value by reference after output fromm model.
    """
    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        #if (batch_idx) % log_interval == 0:
        if batch_idx % 5 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx*5) + ((epoch-1)*len(train_loader.dataset)))


def test(network, test_loader, test_losses):
    """
    A function to test the network and give accuracy and update loss counter.

    Input: network, test_loader, test_losses
    Output: No return; modifies test counter and loss value by reference after output from model.
    """
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
          output = network(data)
          test_loss += F.nll_loss(output, target, size_average=False).item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def epoch_result_plot(epoch, train_counter, train_losses, test_counter, test_losses):
    """
    A function to plot training and testing losses against the number of examples trained.

    Input: epoch, train_counter, train_losses, test_counter, test_losses
    Output: No return; plots the loss against number of training examples.
    """
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.title('Epoch {}'.format(epoch))
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.grid()
    plt.show(block=False)
    plt.pause(0.2) # pause for the plot to be displayed


def plot_six_examples(loader):
    examples = enumerate(loader)

    # Define class names for Greek dataset
    class_names = ['alpha', 'beta', 'chi', 'eta', 'gamma', 'sigma']

    batch_idx, (example_data, example_targets) = next(examples)

    fig = plt.figure()
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(class_names[example_targets[i]]))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


# main function (yes, it needs a comment too)
def main(argv):
    """
    The driver function. Initializes the network and iterates over epochs for training and testing.
    """

    # Initialize and load saved model
    network = Net()
    network.load_state_dict(torch.load('/results/model.pth'))  # Load saved model
    print("Network Loaded from '/results/model.pth' ")

    # freezes the parameters for the whole network
    for param in network.parameters():
        param.requires_grad = False

    # Modify the last layer for the Greek letters
        network.fc2 = nn.Linear(50, 6)
        print(network)
        pause()

    # Set requires_grad to True for the weight and bias parameters of the new layer
    network.fc2.weight.requires_grad = True
    network.fc2.bias.requires_grad = True

    # Get dataset for training
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('/files/greek_train/greek_train',
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=5,
        shuffle=True)

    plot_six_examples(greek_train)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # Get dataset for testing
    g_test = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('/files/greek_train/test', transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GreekTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=5,
        shuffle=True)

    # Training the model
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(greek_train) for i in range(n_epochs + 1)]

    if not os.path.exists('/results_greek'):
        os.makedirs('/results_greek')  # Directory check for storing results

    test(network, g_test, test_losses)  # Pre-training check

    for epoch in range(1, n_epochs + 1):  # run for 5 epochs
        train(epoch, network, optimizer, greek_train, train_losses, train_counter)
        test(network, g_test, test_losses)
    epoch_result_plot(epoch, train_counter, train_losses, test_counter[:epoch + 1], test_losses)
    pause()

    # Define class names for Greek dataset
    class_names = ['alpha', 'beta', 'chi', 'eta', 'gamma', 'sigma']

    # Evaluating the model against test data and predictions
    network.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(g_test):
            if batch_idx == 0:
                output = network(data)
                for i in range(len(target)):
                    print('Outputs for example', i + 1)
                    for j in range(5):
                        print('{:.2f}'.format(torch.exp(output[i][j]).item()), end=' ')
                    print('\nPredicted:', class_names[output.data.max(1, keepdim=True)[1][i].item()], 'Actual:',
                          class_names[target[i].item()])
                fig = plt.figure()
                for i in range(5):
                    plt.subplot(3, 3, i + 1)
                    plt.tight_layout()
                    plt.imshow(data[i][0], cmap='gray', interpolation='none')
                    plt.title("Predicted: " + class_names[output.data.max(1, keepdim=True)[1][i].item()])
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
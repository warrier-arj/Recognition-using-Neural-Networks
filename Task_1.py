#
"""
Arjun Rajeev Warrier
CS 5330: Pattern Recognition and Computer Vision
Project 5: Recognition using Deep Networks

Task_1.py trains the model against the MNIST dataset and stores it to an external location while plotting losses.
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
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10


def pause():
    """
    Function whose primary purpose is to pause a program during runtime till the ENTER key is pressed.
    Used to view and debug.

    Input: None
    Output: None
    """
    programPause = input("Press <Enter> key to continue...")


class DataSet:
    """
    Class to load and store the MNIST digit datasets for testing and training.
    Also contains function to plot six examples and check shape.
    """
    # To download dataset
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    def plot_six_examples(self):
        """
            A function to plot the first six examples.

            Input: test_loader form self
            Output: None, subplot of example
            """
        examples = enumerate(self.test_loader)
        batch_idx, (example_data, example_targets) = next(examples)

        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.show()
        #pause()

    def check_data(self):
        # Function to print shape of one test batch
        # Useful for verifying type and size
        examples = enumerate(self.test_loader)
        batch_idx, (example_data, example_targets) = next(examples)
        print(example_data.shape)


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
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
          train_losses.append(loss.item())
          train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
          torch.save(network.state_dict(), '/results/model.pth')
          torch.save(optimizer.state_dict(), '/results/optimizer.pth')


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


# main function
def main(argv):
    """
    The driver function. Initializes the network and iterates over epochs for training and testing.
    """
    # initialize random seed
    random_seed = 42
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Task 1. A,B
    t1 = DataSet()
    t1.plot_six_examples()
    t1.check_data()

    # Task 1 C
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    print(network)
    pause()
    # Task 1 D
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(t1.train_loader.dataset) for i in range(n_epochs + 1)]

    if not os.path.exists('/results'):
        os.makedirs('/results')                              # Directory check for storing results

    test(network, t1.test_loader, test_losses)               # Pre-training check

    for epoch in range(1, n_epochs + 1):                     # run for 5 epochs
        train(epoch, network, optimizer, t1.train_loader, train_losses, train_counter)
        test(network, t1.test_loader, test_losses)
    epoch_result_plot(epoch, train_counter, train_losses, test_counter[:epoch+1], test_losses)
    pause()

    # Task 1 E
    torch.save(network.state_dict(), '/results/model.pth')
    print("Network Saved to '/results/model.pth' ")

    print("End of Program")
    return

if __name__ == "__main__":
    main(sys.argv)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""

End of program

"""
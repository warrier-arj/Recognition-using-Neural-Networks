#
"""
Arjun Rajeev Warrier
CS 5330: Pattern Recognition and Computer Vision
Project 5: Recognition using Deep Networks

Task_4_Final.py trains and tests the model for different combinations of varied hyper params.
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
import time

# Define the hyperparameters to search over and some other global variables
num_conv_layers_values = [2, 3, 4]
conv_filter_size_values = [1, 2, 3]
num_conv_filters_values = [16, 32]
num_epochs = [1, 3, 5]
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
acc_counter = 0


# The dataset
class DataSet:
    """
    Class to load and store the MNIST digit datasets for testing and training.
    Also contains function to plot six examples and check shape.
    """
    def __init__(self):
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST('/files/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST('/files/', train=False, download=True,
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

    def check_data(self):

        # Function to print shape of one test batch
        # Useful for verifying type and size
        examples = enumerate(self.test_loader)
        batch_idx, (example_data, example_targets) = next(examples)
        print(example_data.shape)


# The network
class Net(nn.Module):
    """
    Class to initialize a network model.
    Also contains a forward function to propagate an input through the network
    """
    def __init__(self, num_conv_layers, conv_filter_size, num_conv_filters):
        super(Net, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(1, num_conv_filters, kernel_size=conv_filter_size, padding=1))
        for i in range(num_conv_layers - 1):
            self.conv_layers.append(nn.Conv2d(num_conv_filters, num_conv_filters, kernel_size=conv_filter_size, padding=1))
        self.fc1 = nn.Linear(self.num_flat_features(), 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        """
        A function test a particular input x against the different layers.

        Input: input x
        Output: log_softmax value of final result
        """
        for conv_layer in self.conv_layers:
            x = F.relu(F.max_pool2d(conv_layer(x), 2))
        x = x.view(-1, self.num_flat_features())
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def num_flat_features(self):
        """
        A function to assist in formation of layer and/or max_pooling
        """
        x = torch.randn(1, 1, 28, 28)
        for conv_layer in self.conv_layers:
            x = F.relu(F.max_pool2d(conv_layer(x), 2))
        return int(torch.prod(torch.tensor(x.size())))


def pause():
    # Function whose primary purpose is to pause a program during runtime till any other key is pressed.
    programPause = input("Press <Enter> key to continue...")


def train(epoch, network, optimizer, train_loader, train_losses, train_counter, log_interval, num_conv_layers, conv_filter_size, num_conv_filters):
    """
    A function to train the network using different hyper-params.

    Input: epoch, network, optimizer, train_loader, train_losses, train_counter, log_interval, num_conv_layers, conv_filter_size, num_conv_filters
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
                (batch_idx*len(data)) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), '/results/model.pth')
            torch.save(optimizer.state_dict(), '/results/optimizer.pth')


def test(network, test_loader, test_losses, num_conv_layers, conv_filter_size, num_conv_filters):
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
    global acc_counter
    acc_counter= correct


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
    plt.pause(0.2)


def main(argv):
    """
    The driver function. Initializes the network and iterates over epochs for training and testing.
    """
    counter = 0
    accuracies = {}
    times = {}

    if not os.path.exists('/results'):
        os.makedirs('/results')

    # Nested loop to iterate through various combinations
    for num_epoch in num_epochs:
        for num_conv_layers in num_conv_layers_values:
            for conv_filter_size in conv_filter_size_values:
                for num_conv_filters in num_conv_filters_values:

                    # Record the start time
                    start_time = time.time()

                    print('Training network with num_conv_layers = {}, conv_filter_size = {}, num_conv_filters = {} and epochs = {}'.format(
                        num_conv_layers, conv_filter_size, num_conv_filters, num_epoch))

                    t1 = DataSet()
                    network = Net(num_conv_layers, conv_filter_size, num_conv_filters)
                    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
                    train_losses = []
                    train_counter = []
                    test_losses = []
                    test_counter = [i * len(t1.train_loader.dataset) for i in range(num_epoch + 1)]

                    test(network, t1.test_loader, test_losses, num_conv_layers, conv_filter_size, num_conv_filters)

                    for epoch in range(1, num_epoch + 1):
                        train(epoch, network, optimizer, t1.train_loader, train_losses, train_counter, log_interval,
                              num_conv_layers, conv_filter_size, num_conv_filters)

                        test(network, t1.test_loader, test_losses, num_conv_layers, conv_filter_size, num_conv_filters)

                    # Record the end time
                    end_time = time.time()
                    #epoch_result_plot(epoch, train_counter, train_losses, test_counter[:epoch+1], test_losses)

                    model_filename = '/results/model_{}_{}_{}.pth'.format(num_conv_layers, conv_filter_size, num_conv_filters)
                    torch.save(network.state_dict(), model_filename)
                    print("Network Saved to '{}'".format(model_filename))
                    counter +=1
                    print(" Counter is {}".format(counter))

                    # Compute the time taken and store it in the dictionary
                    key = (num_conv_layers, conv_filter_size, num_conv_filters)
                    times[key] = end_time - start_time

                    # After the model has been trained and tested, compute the accuracy and store it in the dictionary
                    accuracy = 100. * acc_counter / len(t1.test_loader.dataset)
                    key = (num_conv_layers, conv_filter_size, num_conv_filters)
                    accuracies[key] = accuracy
    print("End of Program")

    # Print accuracy and computation times for each combination
    for key, accuracy in accuracies.items():
        time_taken = times[key]
        print("Accuracy for hyperparameters {}: {:.2f}%, Time taken: {:.2f} seconds".format(key, accuracy, time_taken))

    pause()
    return

if __name__ == "__main__":
    main(sys.argv)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""

End of program

"""
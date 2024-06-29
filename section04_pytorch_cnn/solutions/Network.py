import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as dataloader

from tqdm.notebook import trange, tqdm


class LeNet(nn.Module):
    def __init__(self, channels_in, device, loss_fun, batch_size, learning_rate):
        # Call the __init__ function of the parent nn.module class
        super(LeNet, self).__init__()
        # Define Convolution Layers
        # conv1 6 channels_inx5x5 kernels
        self.optimizer = None
        self.conv1 = nn.Conv2d(channels_in, 6, kernel_size=5)
        
        # conv2 16 6x5x5 kernels
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Define MaxPooling Layers
        # Default Stride is = to kernel_size
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Define Linear/Fully connected/ Dense Layers
        # Input to linear1 is the number of features from previous conv - 16x5x5
        # output of linear1 is 120
        self.linear1 = nn.Linear(16*5*5, 120)
        # output of linear2 is 84
        self.linear2 = nn.Linear(120, 84)
        # output of linear3 is 10
        self.linear3 = nn.Linear(84, 10)

        self.device = device
        self.loss_fun = loss_fun
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.train_loss_logger = []

        self.train_acc_logger = []
        self.val_acc_logger = []

        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None

        self.set_optimizer()

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def set_data(self, train_set, test_set, val_set):

        print(f'Number of training examples: {len(train_set)}')
        print(f'Number of validation examples: {len(val_set)}')
        print(f'Number of testing examples: {len(test_set)}')

        self.train_loader = dataloader.DataLoader(train_set, shuffle=True, batch_size=self.batch_size, num_workers=4)
        self.valid_loader = dataloader.DataLoader(val_set, batch_size=self.batch_size, num_workers=4)
        self.test_loader = dataloader.DataLoader(test_set, batch_size=self.batch_size, num_workers=4)

    # This function should perform a single training epoch using our training data
    def train_model(self):

        if self.train_loader is None:
            ValueError("Dataset not defined!")

        # Set Network in train mode
        self.train()
        for i, (x, y) in enumerate(tqdm(self.train_loader, leave=False, desc="Training")):
            # Forward pass of image through network and get output
            fx = self.forward(x.to(self.device))

            # Calculate loss using loss function
            loss = self.loss_fun(fx, y.to(self.device))

            # Zero gradients
            self.optimizer.zero_grad()
            # Backpropagate gradients
            loss.backward()
            # Do a single optimization step
            self.optimizer.step()

            # Log the loss for plotting
            self.train_loss_logger.append(loss.item())

    # This function should perform a single evaluation epoch, it WILL NOT be used to train our model
    def evaluate_model(self, train_test_val="test"):
        if self.test_loader is None:
            ValueError("Dataset not defined!")

        state = "Evaluating "
        if train_test_val == "test":
            loader = self.test_loader
            state += "Test Set"
        elif train_test_val == "train":
            loader = self.train_loader
            state += "Train Set"
        elif train_test_val == "val":
            loader = self.valid_loader
            state += "Validation Set"
        else:
            ValueError("Invalid dataset, train_test_val should be train/test/val")

        # Initialise counter
        epoch_acc = 0
        self.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(loader, leave=False, desc=state)):
                # Forward pass of image through network
                fx = self.forward(x.to(self.device))

                # Log the cumulative sum of the acc
                epoch_acc += (fx.argmax(1) == y.to(self.device)).sum().item()

        # Log the accuracy from the epoch
        if train_test_val == "train":
            self.train_acc_logger.append(epoch_acc / len(loader.dataset))
        elif train_test_val == "val":
            self.val_acc_logger.append(epoch_acc / len(loader.dataset))

        return epoch_acc / len(loader.dataset)

    def forward(self, x):
        # Pass input through conv layers
        # x shape is BatchSize-3-32-32
        
        out1 = F.relu(self.conv1(x))
        # out1 shape is BatchSize-6-28-28
        out1 = self.maxpool(out1)
        # out1 shape is BatchSize-6-14-14

        out2 = F.relu(self.conv2(out1))
        # out2 shape is BatchSize-16-10-10
        out2 = self.maxpool(out2)
        # out2 shape is BatchSize-16-5-5

        # Flatten out2 to shape BatchSize-16x5x5
        out2 = out2.view(out2.shape[0], -1)
        
        out3 = F.relu(self.linear1(out2))
        # out3 shape is BatchSize-120
        out4 = F.relu(self.linear2(out3))
        # out4 shape is BatchSize-84
        out5 = self.linear3(out4)
        # out5 shape is BatchSize-10
        return out5
    
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
import os

from tqdm.notebook import trange, tqdm


class ModelTrainer(nn.Module):
    def __init__(self, model, device, loss_fun, batch_size, learning_rate, save_dir, model_name, start_from_checkpoint=False):
        # Call the __init__ function of the parent nn.module class
        super(ModelTrainer, self).__init__()
        self.optimizer = None
        self.model = model

        self.device = device
        self.loss_fun = loss_fun
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_epoch = 0
        self.best_valid_acc = 0

        self.train_loss_logger = []
        self.train_acc_logger = []
        self.val_acc_logger = []

        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None

        self.set_optimizer()
        self.save_path = os.path.join(save_dir, model_name + ".pt")
        self.save_dir = save_dir

        # Create Save Path from save_dir and model_name, we will save and load our checkpoint here
        # Create the save directory if it does note exist
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        if start_from_checkpoint:
            self.load_checkpoint()
        else:
            # If checkpoint does exist and start_from_checkpoint = False
            # Raise an error to prevent accidental overwriting
            if os.path.isfile(self.save_path):
                raise ValueError("Warning Checkpoint exists")
            else:
                print("Starting from scratch")

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def load_checkpoint(self):
        # Check if checkpoint exists
        if os.path.isfile(self.save_path):
            # Load Checkpoint
            check_point = torch.load(self.save_path)

            # Checkpoint is saved as a python dictionary
            # Here we unpack the dictionary to get our previous training states
            self.model.load_state_dict(check_point['model_state_dict'])
            self.optimizer.load_state_dict(check_point['optimizer_state_dict'])

            self.start_epoch = check_point['epoch']
            self.best_valid_acc = check_point['best_valid_acc']

            self.train_loss_logger = check_point['train_loss_logger']
            self.train_acc_logger = check_point['train_acc_logger']
            self.val_acc_logger = check_point['val_acc_logger']

            print("Checkpoint loaded, starting from epoch:", self.start_epoch)
        else:
            # Raise Error if it does not exist
            raise ValueError("Checkpoint Does not exist")

    def save_checkpoint(self, epoch, valid_acc):
        self.best_valid_acc = valid_acc

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_valid_acc': valid_acc,
            'train_loss_logger': self.train_loss_logger,
            'train_acc_logger': self.train_acc_logger,
            'val_acc_logger': self.val_acc_logger,
        }, self.save_path)

    def set_data(self, train_set, test_set, val_set):

        print(f'Number of training examples: {len(train_set)}')
        print(f'Number of validation examples: {len(val_set)}')
        print(f'Number of testing examples: {len(test_set)}')

        self.train_loader = dataloader.DataLoader(train_set, shuffle=True, batch_size=self.batch_size, num_workers=4)
        self.valid_loader = dataloader.DataLoader(val_set, shuffle=False, batch_size=self.batch_size, num_workers=4)
        self.test_loader = dataloader.DataLoader(test_set, shuffle=False, batch_size=self.batch_size, num_workers=4)

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
        return self.model(x)
    
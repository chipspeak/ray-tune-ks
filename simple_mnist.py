# Import PyTorch libraries for building and training neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F  # (Not used here, but often for extra layers)
from torch.utils.data import DataLoader  # Helps load data in batches
from torchvision import datasets, transforms  # Utilities to load and transform datasets

# Import Ray libraries for distributed hyperparameter tuning
from ray import tune
from ray.air import session  # Used to report results from training to Ray Tune (we need session because this runs as a Ray Job)

# ---------------------------
# Define a very simple model
# ---------------------------
'''
A neural network is a mathematical function that can learn patterns in data
Here we define a very simple, one-layer neural network
'''
class Net(nn.Module):  # Inherit from PyTorch's Module class
    def __init__(self):
        super().__init__()  # Call parent constructor
        # Define a single fully connected layer:
        # Takes 784 input values (28x28 pixels from the image), outputs 10 values (one per clothing class)
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        # "Flatten" the input image from 2D (28x28) to a 1D vector (784 values)
        x = x.view(-1, 28 * 28)
        # Pass it through the fully connected layer to get predictions
        return self.fc(x)


# ----------------------------
# Load the training data
# ----------------------------

# This function prepares the data used to train the model
def get_dataloader(batch_size):
    # Transform the images into tensors (PyTorch's native format)
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Download and load the FashionMNIST dataset (images of clothes, like shirts or shoes)
    train_dataset = datasets.FashionMNIST(
        root="/tmp/data",          # Where to store the downloaded data
        train=True,                # Use training split of the dataset
        download=True,             # Download if not already present
        transform=transform        # Apply transformation to each image
    )
    
    # Organize the data into batches and shuffle them
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ----------------------------
# Train the model
# ----------------------------

# This is the main training function that will run inside Ray Tune
def train_mnist(config):
    # Create an instance of our model
    model = Net()
    
    # Create an optimizer — adjusts model weights to minimize the error
    # 'lr' (learning rate) comes from the tuning config
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    
    # Define how we calculate the difference between predicted and actual labels
    loss_fn = nn.CrossEntropyLoss()

    # Load the training data using the specified batch size
    dataloader = get_dataloader(config["batch_size"])

    # Training loop — repeat for a few "epochs" (passes over all the training data)
    for epoch in range(5):  # Keep it short for demonstration
        total_loss = 0
        for x, y in dataloader:        # x = images, y = labels
            optimizer.zero_grad()      # Clear out previous gradients
            out = model(x)             # Run model on current batch
            loss = loss_fn(out, y)     # Compare prediction vs actual answer
            loss.backward()            # Compute gradients
            optimizer.step()           # Update model weights
            total_loss += loss.item()  # Track total loss for reporting
        avg_loss = total_loss / len(dataloader)
        
        # Send this epoch's loss back to Ray Tune
        session.report({"epoch": epoch, "loss": avg_loss})

# ---------------------------------------------
# Use Ray Tune to try different hyperparameters
# ---------------------------------------------

'''
Run a tuning job where Ray will:
- Call the train_mnist() function
- Try different values for learning rate and batch size
- Report how the model performs for each configuration
'''
tune.run(
    train_mnist,
    config={
        # Try 2 learning rates: 0.01 and 0.1 (grid_search means try each one)
        "lr": tune.grid_search([0.01, 0.1]),
        
        # Try 2 batch sizes: 32 and 64 (choose randomly one per trial)
        "batch_size": tune.choice([32, 64])
    },
    num_samples=1,  # Run one training trial per config (grid × choice)
    
    # Specify that each trial uses 1 CPU
    resources_per_trial={"cpu": 1}
)

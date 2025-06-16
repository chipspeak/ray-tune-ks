# Import PyTorch libraries for building and training neural networks
import torch
import torch.nn as nn
import torch.nn.functional as F  # (Not used here, but often for extra layers)
from torch.utils.data import DataLoader  # Helps load data in batches
from torchvision import datasets, transforms  # Utilities to load and transform datasets

# Import Ray libraries for distributed hyperparameter tuning
from ray import tune
from ray.air import session  # Used to report results from training to Ray Tune (we need session because this runs as a Ray Job)

# ------------------------------
# Define a simple neural network
# ------------------------------

'''
A neural network is a mathematical function that can learn patterns in data.
Here we define a slightly deeper network with one hidden layer to improve accuracy.
Very lightweight and suitable for CPU-only systems.
'''
class Net(nn.Module):  # Inherit from PyTorch's Module class
    def __init__(self):
        super().__init__()  # Call parent constructor
        # Fully connected layers:
        self.fc1 = nn.Linear(28 * 28, 128)  # Hidden layer: from 784 inputs to 128 units
        self.fc2 = nn.Linear(128, 10)       # Output layer: from 128 to 10 classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)     # Flatten input image from 2D to 1D
        x = F.relu(self.fc1(x))     # Apply ReLU activation to hidden layer
        return self.fc2(x)          # Output logits (scores for each class)

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
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2              # Use multiple subprocesses to load data faster
    )

# ----------------------------
# Train the model
# ----------------------------

# This is the main training function that will run inside Ray Tune
def train_mnist(config):
    # Set device: use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of our model and move it to the appropriate device
    model = Net().to(device)

    # Set the model to training mode (enables dropout, batchnorm if used)
    model.train()
    
    # Use Adam optimizer — trying this to improve performance
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    # Define how we calculate the difference between predicted and actual labels
    loss_fn = nn.CrossEntropyLoss()

    # Load the training data using the specified batch size
    dataloader = get_dataloader(config["batch_size"])

    # Training loop — repeat for a few "epochs" (passes over all the training data)
    for epoch in range(20): 
        total_loss = 0
        for x, y in dataloader:            # x = images, y = labels
            x, y = x.to(device), y.to(device)  # Move data to the same device as model
            optimizer.zero_grad()          # Clear out previous gradients
            out = model(x)                 # Run model on current batch
            loss = loss_fn(out, y)         # Compare prediction vs actual answer
            loss.backward()                # Compute gradients
            optimizer.step()               # Update model weights
            total_loss += loss.item()      # Track total loss for reporting

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
    num_samples=5,  # Run 5 training trials total, each with a different sampled config
    
    # Specify that each trial uses 1 CPU (or 1 GPU if available)
    resources_per_trial={"cpu": 1}
)

from ucimlrepo import fetch_ucirepo 
import pandas as pd 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 

# reducing the number of columns because n_entries will be too high and sooo demanding.
X = X.copy()  # Ensures X is a separate copy before modifying
X.drop(columns=['fnlwgt', 'education', 'race', 'native-country'], inplace=True)


# Ensure y is a Series (extract first column if it's a DataFrame)
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]  # Take the first column

# Now clean the labels
y = y.astype(str).str.strip().replace({
    '>50K.': '>50K',
    '<=50K.': '<=50K'
})

# # Convert to binary labels
y = y.map({'<=50K': 0, '>50K': 1})


# one hot encoding
X = pd.get_dummies(X)
#y = pd.get_dummies(y)


# normalization
norm = StandardScaler()
X = norm.fit_transform(X)

# 80 % training , 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 2)

# entries of nn (the number of columns of X_train)

n_entries = X_train.shape[1]  


# Convert to NumPy arrays
# X_train = X_train.to_numpy()
# X_test = X_test.to_numpy()

y_train = y_train.to_numpy().reshape(-1, 1)  # Ensure it's 2D
y_test = y_test.to_numpy().reshape(-1, 1)    # Ensure it's 2D

# Convert to PyTorch tensors
t_X_train = torch.from_numpy(X_train).float().to("cpu")
t_X_test = torch.from_numpy(X_test).float().to("cpu")
t_y_train = torch.from_numpy(y_train).float().to("cpu")
t_y_test = torch.from_numpy(y_test).float().to("cpu")

# 3-layer NN (44 → [64 → 32 → 1])

class Network(nn.Module):
    def __init__(self, n_entries):
        super(Network, self).__init__() #super().__init__() --> fancy way
        self.linear1 = nn.Linear(n_entries, 64)  
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        pred_1 = self.relu(self.linear1(inputs)) 
        pred_2 = self.relu(self.linear2(pred_1))  
        y_pred = self.sigmoid(self.linear3(pred_2))  

        return y_pred


#hyperparameters

lr = 1e-3
epochs = 5000
status = 200
# --

model = Network(n_entries=n_entries)
loss_fn = nn.BCELoss() # Binary Cross Entropy
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
print("Model's Architecture{}".format(model))
history = pd.DataFrame()

print("training")
for epoch in range(epochs):
    y_pred = model(t_X_train)
    
    loss = loss_fn(input=y_pred , target=t_y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
    if epoch % status ==0:
        print(f"\nEpoch {epoch} \t Loss: {round(loss.item(), 4)}")
    
    with torch.no_grad():
        y_pred = model(t_X_test)  # Forward pass to get predictions
        y_pred_class = (y_pred > 0.5).float()  # Convert probabilities to binary (0 or 1)
        
        correct = (y_pred_class == t_y_test).float()  # Compare predictions with true labels
        accuracy = correct.sum() / correct.size(0)  # Calculate accuracy

        if epoch % status == 0:
            print(f"Accuracy: {accuracy.item():.4f}")

    
    df_tmp = pd.DataFrame(data={
        'epoch': epoch,
        'loss': round(loss.item(), 4),
        'accuracy': round(accuracy.item(), 4)
    }, index=[0])
    history= pd.concat(objs=[history, df_tmp], ignore_index=True, sort=False)
    
print("accuracy: {}".format(round(accuracy.item(), 4)))

import matplotlib.pyplot as plt
def plot_metrics(history):

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Loss
    axs[0].plot(history['epoch'], history['loss'], label='Loss', color='blue')
    axs[0].set_title("Loss", fontsize=14)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True)

    # Accuracy
    axs[1].plot(history['epoch'], history['accuracy'], label='Accuracy', color='green')
    axs[1].set_title("Accuracy", fontsize=14)
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

plot_metrics(history)

# ------------------------------
# example predict
# ------------------------------

def predict_example(model, X_test, y_test, index=5):
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(X_test[index].unsqueeze(0))).item()
    
    print(f"Predicted Probability: {pred:.4f}")
    print(f"Actual Label: {y_test[index].item()}")

predict_example(model, t_X_test, t_y_test)

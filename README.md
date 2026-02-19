# Developing a Neural Network Regression Model
## NAME: MOHAMED AAKIF ASRAR S
## REG NO:212223240088
## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This code builds and trains a feedforward neural network in PyTorch for a regression task.
The model takes a single input feature, passes it through two hidden layers with ReLU activation, and predicts one continuous output.
It uses MSE loss and RMSProp optimizer to minimize the error between predictions and actual values over training epochs.

## Neural Network Model

<img width="930" height="643" alt="image" src="https://github.com/user-attachments/assets/eaebc717-17d1-4762-89a3-b584d2b05979" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:MOHAMED AAKIF ASRAR S
### Register Number: 212223240088
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def aakif():
    print("Name: MOHAMED AAKIF ASRAR S")
    print("Register Number: 212223240088")

dataset1 = pd.read_csv('/content/DL-Exp1 - Sheet1 (1).csv')

X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 6)
        self.fc3 = nn.Linear(6, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
        aakif()
        print("Neural Network Regression Model Initialized")

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

ai_aakif = NeuralNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(ai_aakif.parameters(), lr=0.01)

def train_model(ai_rash, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_rash(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        ai_rash.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_aakif, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_aakif(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_aakif.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_aakif(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()

aakif()
print(f'Prediction for input 9: {prediction}')


```
## Dataset Information

<img width="167" height="416" alt="image" src="https://github.com/user-attachments/assets/baa37749-11a8-4b51-8df4-7ac7c81ed7a5" />


## OUTPUT

<img width="426" height="296" alt="image" src="https://github.com/user-attachments/assets/5ee4f5d0-46db-485f-b806-17ad3a0b1aac" />


### Training Loss Vs Iteration Plot
<img width="691" height="507" alt="image" src="https://github.com/user-attachments/assets/e6e5e637-46c9-4133-9110-6a8cb7e18606" />



### New Sample Data Prediction

<img width="371" height="73" alt="image" src="https://github.com/user-attachments/assets/a244c604-6700-42f7-a423-a57f24345e6b" />


## RESULT

Successfully executed the code to develop a neural network regression model.


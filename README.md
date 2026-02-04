# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

1. The problem involves building a neural network model to predict continuous numerical values rather than categories.
2. A dataset with input features and corresponding target values is required for training.
3. The data must be preprocessed, including cleaning, normalization, and splitting into training and testing sets.
4. A suitable neural network architecture with input, hidden, and output layers is designed.
5. The model learns patterns by adjusting weights using a loss function like Mean Squared Error.
6. Training is performed using an optimization algorithm such as gradient descent or Adam.
7. Finally, the model is evaluated on test data to measure prediction accuracy and generalization.

## Neural Network Model

Include the neural network model diagram.

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
### Name: Mohamed Aakif Asrar S
### Register Number: 212223240088
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here



```
## Dataset Information

Include screenshot of the dataset

## OUTPUT

### Training Loss Vs Iteration Plot

Include your plot here

### New Sample Data Prediction

Include your sample input and output here

## RESULT

Include your result here

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# Linear Regression Formula: y = mx + b

# bias = 0.3 
# weight = 0.7 # "Slope"

# X = torch.arange(0,1,0.02).unsqueeze(dim=1);
# Y = weight * X + bias

# training_set_bounds = 0.8

# X_train = X[:int(len(X)*0.8)]
# Y_train = Y[:int(len(X)*0.8)]
# X_test = X[int(len(X)*0.8):]
# Y_test = Y[int(len(X)*0.8):]


# class LinearRegressionModel(nn.Module): # <- Almost everything in pytorch inherits nn.Module, It's kind of the building blocks of pytorch.
#     def __init__(self):
#         super().__init__()
#         self.weights= nn.Parameter(torch.randn(1, requires_grad=True,dtype=torch.float))
#         self.bias = nn.Parameter(torch.randn(1, dtype=torch.float))


#     def forward(self, x: torch.Tensor):
#         return self.weights * x + self.bias


# torch.manual_seed(42)
# model_0 = LinearRegressionModel()

# # Setting up a loss function
# MAE_fn = nn.L1Loss() # "Mean absolute error" function. Fancy math!

# # Setting up an optimizer to work with the loss function to get the parameters closer and closer to their actual value
# optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.0001) # Stochastic Gradient Descent, lr = "learning rate", how much it changes the parameter per "step"

# # An epoch is one loop through the data
# epochs = 100000
# print(model_0.state_dict())

# # Loop through the data... 

# for epoch in range(epochs):
#     model_0.train()
    
#     y_pred = model_0(X_train)
    
#     loss = MAE_fn(y_pred, Y_train)
   
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     model_0.eval()
    
#     ### Testing ###
#     with torch.inference_mode():
#         test_preds = model_0(X_test)
#         test_loss = MAE_fn(test_preds, Y_test)
        
#     if (epoch % 10000 == 0):
#         print(f"Loop: {epoch} | Loss: {loss} | Test loss: {test_loss}")
    
# torch.save(model_0.state_dict(), "./model_0")
# print(model_0.state_dict())

# model_0 = LinearRegressionModel()
# model_0.load_state_dict(torch.load("./model_0"))

# with torch.inference_mode():
#     y_preds = model_0(X_test)
#     plot_predictions(X_train, Y_train, X_test, Y_test, y_preds)

weight = 0.1
bias = 0.3

training_data = torch.arange(0,1,0.002).unsqueeze(dim=1)
y = weight * training_data + bias

train_data = training_data[:int(0.8*len(training_data))]
train_labels = y[:int(0.8*len(y))]
test_data = training_data[int(0.8*len(training_data)):]
test_labels = y[int(0.8*len(y)):]

def plot_predictions(train_data, train_label, test_data, test_label, predictions = None):
    plt.figure(figsize=(10,7))
    
    # "Plot" our train data in blue
    plt.scatter(train_data, train_label, c="b", s=4, label="Training Data")

    # "Plot" our test data in red
    plt.scatter(test_data, test_label, c="r", s=4, label="Test Data")
    
    # If there's a prediction, plot it in green
    if (predictions is not None):
        plt.scatter(test_data, predictions, c="g", s=4, label="Predictions")

    # Show the label legends
    plt.legend(prop={"size": 14})
    
    plt.show() 
    
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,out_features=1  )
        
    def forward(self, x):
        return self.linear_layer(x);
    
torch.manual_seed(42)
model_1 = LinearRegressionModelV2();
model_1.to("cuda")  

loss_fn = nn.L1Loss();
optimzer = torch.optim.SGD(params=model_1.parameters(), lr=0.0001);

epochs = 100000

train_data = train_data.to("cuda")
train_labels = train_labels.to("cuda")
test_data = test_data.to("cuda")
test_labels = test_labels.to("cuda")


for epoch in range(epochs+1):
    model_1.train()
    
    y_pred = model_1(train_data)
    loss = loss_fn(y_pred, train_labels)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    model_1.eval()

    with torch.inference_mode():
        test_preds = model_1(test_data)
        test_loss = loss_fn(test_preds, test_labels)
    if (epoch % 10000 == 0):
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")

with torch.inference_mode():
    test_preds = model_1(test_data)
    plot_predictions(train_data.cpu(), train_labels.cpu(), test_data.cpu(), test_labels.cpu(), test_preds.cpu())    
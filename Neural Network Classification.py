from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn

x, y = make_circles(n_samples=1000, noise=0.05, random_state=42)

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

x_train, x_test ,y_train ,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

device = "cuda" if torch.cuda.is_available() else "cpu"

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=2, out_features=128),nn.ReLU() ,nn.Linear(in_features=128, out_features=1))

    def forward(self, x):
        return self.layers(x)
    
model_0 = CircleModel().to(device)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

def accuracy(true, test):
    correct = torch.eq(true,test).sum().item()
    acc = (correct/len(test)) * 100
    return acc

epochs = 100000

torch.manual_seed(42)
torch.cuda.manual_seed(42)
for epoch in range(epochs+1):
    model_0.train()
    
    y_logits = model_0(x_train.to(device)).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    loss = loss_fn(y_logits, y_train.to(device).float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model_0.eval()
    if epoch % 10000 == 0:
        with torch.inference_mode():
            test_logits = model_0(x_test.to(device)).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            
            test_acc = accuracy(y_test.to(device), test_pred)
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Accuracy: {test_acc:.2f}%")
    
    



    
    
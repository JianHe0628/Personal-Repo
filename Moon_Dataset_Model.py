import torch
import torch.nn as nn
import sklearn as skt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

#Create Dataset and Convert to tensors
X,y = make_moons(10000,shuffle=True,noise=None)
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

X_train,X_split,Y_train,Y_split = train_test_split(X,y,test_size=0.2,random_state=42)
X_train,Y_train = X_train.to(device),Y_train.to(device)
split_value = int(len(Y_split)*0.5)
X_eval,X_test = X_split[:split_value].to(device),X_split[split_value:].to(device)
Y_eval,Y_test = Y_split[:split_value].to(device),Y_split[split_value:].to(device)

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.
    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.winter)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def accuracy_calc(data,pred):
    return ((torch.eq(data,pred).sum().item())/len(data))*100

class Classification_Model(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        # self.layer1 = nn.Linear(in_features=input,out_features=10)
        # self.layer2 = nn.Linear(in_features=10,out_features=10)
        # self.layer3 = nn.Linear(in_features=10,out_features=output)
        # self.Relu = nn.ReLU()

        self.layers = nn.Sequential(nn.Linear(in_features=input,out_features=10),
                                    nn.ReLU(),
                                    nn.Linear(in_features=10,out_features=10),
                                    nn.ReLU(),
                                    nn.Linear(in_features=10,out_features=output))
    def forward(self,data):
        return self.layers(data)
    

##Training Loop
Epochs = 1000
#Declare Model with 
Model_Obj = Classification_Model(2,1).to(device)
print(Model_Obj)
#Define Loss Function and Optimizer
Loss_Func = nn.BCEWithLogitsLoss()
Optim = torch.optim.SGD(Model_Obj.parameters(),lr=0.1)


for x in range(Epochs):
    Model_Obj.train()
    y_train_logits = Model_Obj(X_train).squeeze()
    y_train_pred = torch.round(torch.sigmoid(y_train_logits))


    train_acc = accuracy_calc(Y_train,y_train_pred)
    train_loss = Loss_Func(y_train_logits,Y_train)

    Optim.zero_grad()

    train_loss.backward()

    Optim.step()

    #Evaluation
    Model_Obj.eval()
    with torch.inference_mode():
        y_eval_logits = Model_Obj(X_eval).squeeze()
        y_eval_pred = torch.round(torch.sigmoid(y_eval_logits))

        eval_acc = accuracy_calc(Y_eval,y_eval_pred)
        eval_loss = Loss_Func(y_eval_logits,Y_eval)


    if x % 100 == 0:
        print(f'Epoch: {x}| Train Loss:   {train_loss:.2f}| Train Accuracy:   {train_acc:.2f}')
        print(f'Epoch: {x}| Eval Loss:   {eval_loss:.2f}|   Eval Accuracy:   {eval_acc:.2f}')

plot_decision_boundary(Model_Obj,X_eval,Y_eval)




import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchmetrics import Precision
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else "cpu"


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

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

torch.manual_seed(42)
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)
X_train,X_Eval,Y_train,Y_Eval = train_test_split(X,y,test_size=0.2,random_state=42)
X_train,X_Eval,Y_train,Y_Eval = X_train.to(device),X_Eval.to(device),Y_train.to(device),Y_Eval.to(device)

class Classifier_Model(nn.Module):
    def __init__(self,inputs,outputs,hneurons):
        super().__init__()
        self.layers = nn.Sequential(
            #nn.ReLU(),
            nn.Linear(in_features=inputs,out_features=hneurons),
            nn.ReLU(),
            nn.Linear(in_features=hneurons,out_features=hneurons),
            nn.ReLU(),
            nn.Linear(in_features=hneurons,out_features=hneurons),
            nn.ReLU(),
            nn.Linear(in_features=hneurons,out_features=outputs)
        )
    
    def forward(self,data):
       return self.layers(data)

Model_Obj0 = Classifier_Model(2,10,3).to(device)

Loss_Func = nn.CrossEntropyLoss()
Optim = torch.optim.Adam(Model_Obj0.parameters(),lr=0.01)
Epoch = 20000

def accuracy_func(actual,predicted):
   return (torch.eq(actual,predicted).sum()/len(actual))*100

##Alternative Accuracy
Accuracy_func2 = Accuracy(task='multiclass',num_classes=3).to(device)
prec_func = Precision(task='multiclass',num_classes=3).to(device)

torch.manual_seed(42)
for x in range(Epoch):
    Model_Obj0.train()
    Y_Train_Logits = Model_Obj0(X_train)
    Y_Train_Preds = torch.softmax(Y_Train_Logits,dim=1).argmax(dim=1)

    Train_Acc = Accuracy_func2(Y_Train_Preds,Y_train)
    Train_Prec = prec_func(Y_Train_Preds,Y_train)
    Y_train = Y_train.type(torch.LongTensor).to(device) ###Since Logits is outputed as Long Tensor, The original Train data will also need to be changed

    Train_Loss = Loss_Func(Y_Train_Logits,Y_train)

    Optim.zero_grad()

    Train_Loss.backward()

    Optim.step()

    #Evaluation Mode
    Model_Obj0.eval()
    with torch.inference_mode():
        Y_Eval_Logits = Model_Obj0(X_Eval)
        Y_Eval_Preds = torch.softmax(Y_Eval_Logits,dim=1).argmax(dim=1)
        Y_Eval = Y_Eval.type(torch.LongTensor).to(device)

        Eval_Loss = Loss_Func(Y_Eval_Logits,Y_Eval)
        Eval_Prec = prec_func(Y_Eval_Preds,Y_Eval)
        Eval_Acc = Accuracy_func2(Y_Eval_Preds,Y_Eval)
    if x%1000 == 0:
        print(f'Epoch:  {x}|Train Loss: {Train_Loss:.4f}|Train Accuracy:  {Train_Acc:.4f}|Precision:  {Train_Prec:.4f}')
        print(f'Epoch:  {x}|Eval Loss:  {Eval_Loss:.4f}|Eval Accuracy:   {Eval_Acc:.4f}|Precision:   {Eval_Prec:.4f}')

plot_decision_boundary(Model_Obj0,X_Eval,Y_Eval)
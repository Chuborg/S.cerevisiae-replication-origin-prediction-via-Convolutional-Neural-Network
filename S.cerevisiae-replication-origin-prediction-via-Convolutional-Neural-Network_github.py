# Ordinal encoding
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import style
%matplotlib inline

# Use GPU
torch.cuda.is_available()
device = torch.device("cuda:0")
print(device)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# Load preprocessed data from .npy file
training_data = np.load("D:/Projects/ORI/hybrid_train_1.5.npy", allow_pickle=True)
testing_1_data = np.load("D:/Projects/ORI/hybrid_test_1_1.5.npy", allow_pickle=True)
testing_2_data = np.load("D:/Projects/ORI/hybrid_test_2_1.5.npy", allow_pickle=True)

print("Number of training samples: ", len(training_data))
print("Number of testing samples: ", len(testing_1_data))
print("Number of testing samples: ", len(testing_2_data))
print("Data prepared: ([ordinal], [label])")

# DEFINE HYPERPARAMETERS
DATA_DIMENTION_ORD = 32 # Data dimension for ordinal encoding
LEARNING_RATE = 0.07
MOMENTUM = 0.8 # For SGD only
BATCH_SIZE = 128
EPOCHS = 70


# TRANSFORM TO TENSORS
# Ordinal encoding data
ordinal_X = torch.Tensor([i[0] for i in training_data]).view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD)
ordinal_test_1_X = torch.Tensor([i[0] for i in testing_1_data]).view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD)
ordinal_test_2_X = torch.Tensor([i[0] for i in testing_2_data]).view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD)

# Labels
y = torch.Tensor([i[1] for i in training_data])
test_1_y = torch.Tensor([i[1] for i in testing_1_data])
test_2_y = torch.Tensor([i[1] for i in testing_2_data])


print("Data preparation: ")
print("Number of ordinal training samples: ", len(ordinal_X))
print("Number of ordinal test_1 samples: ", len(ordinal_test_1_X))
print("Number of ordinal test_2 samples: ", len(ordinal_test_2_X))
print(" \n")
print("Labels preparation: ")
print("Number of training labels: ", len(y))
print("Number of test_1 labels: ", len(test_1_y))
print("Number of test_2 labels: ", len(test_2_y))


# Conv net. Input for the first fc layer calculated automatically.
class Net(nn.Module):
    def __init__(self):
        global DATA_DIMENTION_ORD
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 128, 5)
        
        x = torch.randn(DATA_DIMENTION_ORD, DATA_DIMENTION_ORD).view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD)
        
        self._to_linear = None
        self.convs(x)        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)
        
    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.relu(self.conv4(x))
        
        if self._to_linear is None:
            self._to_linear = np.prod(x[0].shape) # Easier to remember
        
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return F.softmax(x, dim=1)
    
net = Net().to(device)
print(net)

# Define optimizer
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
loss_function = nn.MSELoss()

VAL_PCT = 0.1
val_size = int(len(ordinal_X) * VAL_PCT)
print("Validation size: ", val_size)

# Define number of training and test samples derived from training data
train_X = ordinal_X[:-val_size]
train_y = y[:-val_size]

test_X = ordinal_X[-val_size:]
test_y = y[-val_size:]

print("Number of training samples: ", len(train_X))
print("Number of test samples: ", len(test_X))

MODEL_NAME = f"model-{int(time.time())}"
print("Model name: ", MODEL_NAME)


def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True) / len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()
    
    return acc, loss


def new_train():
    global DATA_DIMENTION_ORD, BATCH_SIZE, EPOCHS
    losses = []

    with open("D:/Projects/ORI/model.log", "a") as f: # Saving log file
        for epoch in range(EPOCHS):
            
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):                
                batch_X = train_X[i: i + BATCH_SIZE].view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD).to(device)
                batch_y = train_y[i: i + BATCH_SIZE].to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                if i % 100 == 0: # Every N steps calculate accuracy, validation loss
                    val_acc, val_loss = new_test(size=40)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")
                
            print("Epoch: ", epoch, "Loss: ", round(float(loss), 3))
            losses.append(round(float(loss), 3))

    plt.plot(losses)
    plt.legend("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")

def new_test(size=32):
    global DATA_DIMENTION_ORD
    random_start = np.random.randint(len(test_X) - size)
    X, y = test_X[random_start: random_start + size], test_y[random_start: random_start + size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD).to(device), y.to(device))

    return val_acc, val_loss

style.use("ggplot")

model_name = MODEL_NAME


def create_acc_loss_graph(model_name):
    contents = open("D:/Projects/ORI/model.log", "r").read().split("\n") # Path to log file
    
    times = []
    accuracies = []
    losses = []
    
    val_accs = []
    val_losses = []
    
    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss = c.split(",")
            
            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))
            
            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))
            
    fig = plt.figure()
    
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)
    
    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2)
    
    ax2.plot(times, losses, label="loss")
    ax2.plot(times, val_losses, label="val_loss")
    ax2.legend(loc=2)

    plt.xlabel("Sample count")
    plt.ylabel("Function value")
    plt.show()

print("Show figure")

def test(net):
    global DATA_DIMENTION_ORD
    print("begin the test")
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            input1 = test_X[i].view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD)
            input1 = input1.to(device)

            net_out = net(input1)[0]

            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy on test samples: ", round(correct / total, 3))
    
# Accuracy test on custom data
def test_1(net):
    global DATA_DIMENTION_ORD
    print("begin the test 1")
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(ordinal_test_1_X))):
            real_class = torch.argmax(test_1_y[i]).to(device)
            input1 = ordinal_test_1_X[i].view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD)
            input1 = input1.to(device)

            net_out = net(input1)[0]

            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy on test_1 samples: ", round(correct / total, 3))
    
def test_2(net):
    global DATA_DIMENTION_ORD
    print("begin the test 2")
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(ordinal_test_2_X))):
            real_class = torch.argmax(test_2_y[i]).to(device)
            input1 = ordinal_test_2_X[i].view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD)
            input1 = input1.to(device)

            net_out = net(input1)[0]

            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy on test_2 samples: ", round(correct / total, 3))
    

new_train()
test(net)
test_1(net)
test_2(net)
create_acc_loss_graph(model_name)

print("Number of epochs: ", EPOCHS)
print("Batch size: ", BATCH_SIZE)
print("Learning rate: ", LEARNING_RATE)
print("Momentum: ", MOMENTUM)
print("DONE")

# Same testing functions, but with additional metrics
from math import sqrt


def test_with_metrics(net):
    global DATA_DIMENTION_ORD
    print("begin the test")
    correct = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i]).to(device)
            input1 = test_X[i].view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD)
            input1 = input1.to(device)

            net_out = net(input1)[0]

            predicted_class = torch.argmax(net_out)

            #print("Net out: ", int(predicted_class), "Real out: ", int(real_class))
            if predicted_class == real_class == 0:
                TN += 1
                correct += 1
            elif predicted_class == real_class == 1:
                TP += 1
                correct += 1
            elif predicted_class == 1 and real_class == 0:
                FP += 1
            elif predicted_class == 0 and real_class == 1:
                FN += 1
            total += 1
    print("True negative: ", TN, "True positive: ", TP, "False positive: ", FP, "False negative: ", FN)
    print("Total: ", total)
    
    sensitivity = round(TP / (TP+FN), 3)
    specificity = round(TN / (TN+FP), 3)
    MCC = round((TP*TN - FP*FN) / sqrt((TP+FN) * (TN+FN) * (TP+FP) * (TN+FP)), 3)

    print("Test metrics")
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Accuracy: ", round(correct / total, 3))
    print("MCC: ", MCC)

def test_1_with_metrics(net):
    global DATA_DIMENTION_ORD
    print("begin the test 1")
    correct = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(ordinal_test_1_X))):
            real_class = torch.argmax(test_1_y[i]).to(device)
            input1 = ordinal_test_1_X[i].view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD)
            input1 = input1.to(device)

            net_out = net(input1)[0]

            predicted_class = torch.argmax(net_out)

            #print("Net out: ", int(predicted_class), "Real out: ", int(real_class))
            if predicted_class == real_class == 0:
                TN += 1
                correct += 1
            elif predicted_class == real_class == 1:
                TP += 1
                correct += 1
            elif predicted_class == 1 and real_class == 0:
                FP += 1
            elif predicted_class == 0 and real_class == 1:
                FN += 1
            total += 1
    print("True negative: ", TN, "True positive: ", TP, "False positive: ", FP, "False negative: ", FN)
    print("Total: ", total)
    
    sensitivity = round(TP / (TP+FN), 3)
    specificity = round(TN / (TN+FP), 3)
    MCC = round((TP*TN - FP*FN) / sqrt((TP+FN) * (TN+FN) * (TP+FP) * (TN+FP)), 3)

    print("Test 1 metrics")
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Accuracy: ", round(correct / total, 3))
    print("MCC: ", MCC)


def test_2_with_metrics(net):
    global DATA_DIMENTION_ORD
    print("begin the test 2")
    correct = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(ordinal_test_2_X))):
            real_class = torch.argmax(test_2_y[i]).to(device)
            input1 = ordinal_test_2_X[i].view(-1, 1, DATA_DIMENTION_ORD, DATA_DIMENTION_ORD)
            input1 = input1.to(device)

            net_out = net(input1)[0]

            predicted_class = torch.argmax(net_out)

            #print("Net out: ", int(predicted_class), "Real out: ", int(real_class))
            if predicted_class == real_class == 0:
                TN += 1
                correct += 1
            elif predicted_class == real_class == 1:
                TP += 1
                correct += 1
            elif predicted_class == 1 and real_class == 0:
                FP += 1
            elif predicted_class == 0 and real_class == 1:
                FN += 1
            total += 1
    print("True negative: ", TN, "True positive: ", TP, "False positive: ", FP, "False negative: ", FN)
    print("Total: ", total)
    
    sensitivity = round(TP / (TP+FN), 3)
    specificity = round(TN / (TN+FP), 3)
    MCC = round((TP*TN - FP*FN) / sqrt((TP+FN) * (TN+FN) * (TP+FP) * (TN+FP)), 3)

    print("Test 2 metrics")
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Accuracy: ", round(correct / total, 3))
    print("MCC: ", MCC)


test_with_metrics(net)
test_1_with_metrics(net)
test_2_with_metrics(net)
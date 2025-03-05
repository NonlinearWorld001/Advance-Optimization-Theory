import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_data(path, bzs=32):
    # load data
    data = pd.read_excel(path, header=0, names=['y', 'x1', 'x2', 'x3'])
    x0 = data.iloc[:, 1:4].values
    y0 = data.iloc[:, 0].values

    # preprocess data
    x = torch.tensor(x0, dtype=torch.float32)
    y = torch.tensor(y0, dtype=torch.float32).view(-1, 1)

    # create dataset and dataloader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=bzs, shuffle=True)
    return dataloader

# define model
class ANN_model(nn.Module):
    def __init__(self):
        super(ANN_model, self).__init__()
        self.hidden_layer = nn.Linear(3, 4, bias=True)
        self.ReLU = nn.ReLU()
        self.output_layer = nn.Linear(4, 1, bias=True)

    def forward(self, x):
        x_middle = self.hidden_layer(x)
        x_middle = self.ReLU(x_middle)
        x_output = self.output_layer(x_middle)
        return x_output
    

# training function
def train(loader, epochs=1000, lr=0.0001):
    model = ANN_model()
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_model_state = None

    # training
    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        for input, target in loader:
            optimizer.zero_grad()
            outputs = model(input)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            tqdm.write(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

        if loss.item() < best_loss:  
            best_loss = loss.item()  
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    return model

# save model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# load model from file
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# test model and predict
def predict(model, x):
    with torch.no_grad():
        input_x = torch.tensor(x, dtype=torch.float32)
        outputs = model(input_x)
        return outputs.numpy()
    
def main():
    # load data
    dataloader = load_data('assignment-1\Assign1_data.xlsx', bzs=64)

    # train model
    model = train(dataloader, epochs=1000, lr=0.001)

    # save model
    save_model(model,'assignment-1\model-pytorch\\trained_model.pth')

    loaded_model = ANN_model()    
    load_model(loaded_model, 'assignment-1\model-pytorch\\trained_model.pth')  

    print("请输入测试数据（格式：x1 x2 x3，用空格分隔）：")
    while True:  
        try:  
            user_input = input("输入（或输入'q'退出）：")  
            if user_input.lower() == 'q':  
                break  
            sample_input = list(map(float, user_input.strip().split()))  
            if len(sample_input) != 3:  
                print("输入格式错误，请输入3个数值")  
                continue  
            # 预测  
            prediction = predict(loaded_model, [sample_input])  
            print(f"输入: {sample_input}, 预测结果: {prediction[0][0]:.4f}")  
        except ValueError:  
            print("输入格式错误，请输入数值！") 


if __name__ == "__main__":
    main()


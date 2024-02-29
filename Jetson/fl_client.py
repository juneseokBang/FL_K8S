import logging
import socket
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F

class Client(object):
    def __init__(self, dataset):
        self.trainset = dataset
        self.fl_config = None
        self.model = CNNMNIST()

    def set_config(self, fl_config):
        self.fl_config = fl_config
        # logging.info(fl_config.local_ep)

    def set_model(self, model):
        self.model.load_state_dict(model)
        # logging.info(self.fl_config.local_ep)
        self.epochs = self.fl_config.local_ep
        self.batch_size = self.fl_config.local_bs
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.fl_config.lr, momentum=0.5)

    def get_trainloader(self, trainset, batch_size):
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    def set_train(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        trainloader = self.get_trainloader(self.trainset, self.batch_size)
        self.train(self.model, trainloader, self.optimizer, self.epochs, device)

        model_path = 'trained_model.pth'
        torch.save(self.model.state_dict(), model_path)

        url = 'http://192.9.202.7:5000/upload_data'
        pod_ip = socket.gethostbyname(socket.gethostname())
        files = {'file': open(model_path, 'rb')}

        headers = {'X-Pod-IP': pod_ip}
        response = requests.post(url, headers=headers, files=files)
        print(response.text)

    def train(self, model, trainloader, optimizer, epochs, device):
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            for batch_id, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_id % 16 == 0:
                    logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, epochs, loss.item()))
        

class CNNMNIST(nn.Module):
    def __init__(self):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
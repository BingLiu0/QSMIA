import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch.optim as optim
import time
import copy
from torch.optim import lr_scheduler
from tqdm import tqdm


def distill(args, config, model, dataloader):
    print("Distilling  Model:", len(dataloader.dataset))
    teacher_model = model
    student_model = timm.create_model(args.model, pretrained=True, num_classes=config.general.num_classes)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.SGD(student_model.parameters(), lr=config.distill.learning_rate, momentum=config.distill.momentum, weight_decay=config.distill.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.distill.decrease_lr_factor, gamma=config.distill.decrease_lr_every)
    
    since = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    teacher_model.eval()
    student_model.to(device)
    best_model_wts = copy.deepcopy(student_model.state_dict())
    best_loss = float('inf')

    for epoch in tqdm(range(config.distill.epochs), desc="Distilling Progress"):
        student_model.train()  
        running_loss = 0.0

        for data, _ in dataloader:
            inputs = data.to(device)
            optimizer.zero_grad()
                
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
                teacher_soft = torch.softmax(teacher_outputs, dim=1)
            
            student_outputs = student_model(inputs)   
            student_log_soft = torch.log_softmax(student_outputs, dim=1)
            loss = kl_loss(student_log_soft, teacher_soft)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        scheduler.step()
        epoch_loss = running_loss / len(dataloader.dataset)

        if  epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(student_model.state_dict())
        
        tqdm.write(f'Epoch {epoch+1}/{config.distill.epochs} - Distill_Loss: {epoch_loss}')

    time_elapsed = time.time() - since
    print('Distilling complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE Distill")

    student_model.load_state_dict(best_model_wts)
    
    return student_model


class AttackModel(nn.Module):
    def __init__(self, input_dim=384, std=0.15):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 32)
        torch.nn.init.normal_(self.fc1.weight.data, 0, std)
        self.fc2 = nn.Linear(32, 32)
        torch.nn.init.normal_(self.fc2.weight.data, 0, std)
        self.batch_norm = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.normal_(self.fc3.weight.data, 0, std)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.batch_norm(x)
        x = F.relu(self.fc3(x))
        return x


class MLP(nn.Module):
    def __init__(self, dim_in):
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(self.dim_in, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x
    
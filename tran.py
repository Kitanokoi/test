import numpy as np
import torch
from utils.model import ETFlowFormer
from utils.loss import UnsupervisedLoss
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.optim as optim

class LabeledDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class UnlabeledDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_and_split_data(data_path, label_path, labeled_ratio=0.1, train_ratio=0.8, batch_size=32):
    # Load data and labels from .npy files
    data = np.load(data_path)
    labels = np.load(label_path)

    # Create full dataset
    full_dataset = LabeledDataset(data, labels)

    # Calculate sizes for train and test sets
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # Split into train and test
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Calculate size of labeled training set
    labeled_size = int(labeled_ratio * train_size)

    # Create labeled and unlabeled training datasets
    labeled_indices = torch.randperm(len(train_dataset))[:labeled_size]
    labeled_train_dataset = Subset(train_dataset, labeled_indices)

    # Create unlabeled dataset (full training set without labels)
    unlabeled_train_dataset = UnlabeledDataset(train_dataset.dataset.data[train_dataset.indices])

    # Create DataLoaders
    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return labeled_train_loader, unlabeled_train_loader, test_loader

# Example usage
data_path = 'path/to/your/data.npy'
label_path = 'path/to/your/labels.npy'
labeled_ratio = 0.1
batch = 512
epoches = 15
a = 0.5
w = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labeled_train_loader, unlabeled_train_loader, test_loader = load_and_split_data(
    data_path, label_path, labeled_ratio=0.1, train_ratio=0.8, batch_size=batch
)

# shape [bs*pakcetnumbers*d]

d = unlabeled_train_loader[0].shape[-1]
t_len = unlabeled_train_loader[0].shape[1]
labels = 12

mt_flowformer_student = ETFlowFormer(d,d,t_len,labels).to(device)
mt_flowformer_teacher = ETFlowFormer(d,d,t_len,labels).to(device)

l_u = UnsupervisedLoss(a).to(device)
l_s = nn.CrossEntropyLoss()

optimizer_student = optim.Adam(mt_flowformer_student.parameters(), lr=0.001)
optimizer_teacher = optim.Adam(mt_flowformer_teacher.parameters(), lr=0.001)

for epoch in range(epoches):
    mt_flowformer_student.train()
    mt_flowformer_teacher.train()
    
    supervised_loss = l_s(0,0)
    unsupervised_loss = l_u(0,0)
    
    student_loss, teacher_loss, total_loss = 0.0, 0.0, 0.0
    
    for labeled_data, label in labeled_train_loader:
        labeled_data = labeled_data.to(device)
        label = label.to(device)
        
        optimizer_student.zero_grad()
        supervised_predicted_y = mt_flowformer_student(labeled_data).to(device)
        supervised_loss = l_s(supervised_predicted_y,label)
        
        student_loss += supervised_loss.item()
        
        
    for unlabeled_data in unlabeled_train_loader:
        unlabeled_data = unlabeled_data.to(device)
        shuffled_indices = torch.randperm(unlabeled_data.size(0)).to(device)
        shuffled_data = unlabeled_data[shuffled_indices]
        
        augmentated_data = (a*unlabeled_data + (1-a)*shuffled_data).to(device)
        optimizer_teacher.zero_grad()
        
        unsupervised_student = mt_flowformer_student(augmentated_data).to(device)
        unsupervised_teacher = mt_flowformer_teacher(unlabeled_data).to(device)
        
        unsupervised_loss = l_u(unsupervised_teacher,unsupervised_student)
        
        teacher_loss += unsupervised_loss.item()
        
    len_labeled_dataset = len(labeled_train_loader)
    len_unlabeled_dataset = len(unlabeled_train_loader)
        
    final_loss = (unsupervised_loss/len_unlabeled_dataset)*w + supervised_loss/len_labeled_dataset
    final_loss.backward()
    optimizer_student.step()
    optimizer_teacher.step()
    
    total_loss = final_loss.item()
    
    print(f'Epoch {epoch}, Average Loss: {total_loss:.4f}, Student Loss:{student_loss:.4f}, Teacher Loss:{teacher_loss:.4f}')

torch.save(mt_flowformer_student, "sudent_model.pt")
torch.save(mt_flowformer_teacher, "teacher_model.pt")
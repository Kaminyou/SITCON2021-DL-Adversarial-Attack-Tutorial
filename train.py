from src.utils import get_mapping_dict
from src.dataset import ImageNetteDataset
from src.model import get_model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # basic setting
    parser.add_argument('--model_name', type=str, default="resnet18", help='model_name')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--batch_size', type=int, default=50, help="batch_size")
    parser.add_argument('--epochs', type=int, default=3, help="epochs")

    args = parser.parse_args()
    print(f"TRAIN {args.model_name}")

    mapping_folder_to_name, mapping_folder_to_label, mapping_name_to_label, mapping_label_to_name = get_mapping_dict()

    train_dataset = ImageNetteDataset(data_root="./data/imagenette2/train", mapping_folder_to_label=mapping_folder_to_label, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = ImageNetteDataset(data_root="./data/imagenette2/val", mapping_folder_to_label=mapping_folder_to_label, train=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00001)

    model.to(args.device)
    best_val_acc = 0
    for epoch in range(args.epochs):
        train_correct = 0
        train_loss = 0
        train_data_num = 0
        val_correct = 0
        val_loss = 0
        val_data_num = 0
        
        model.train()
        for idx, (data, labels) in enumerate(train_dataloader):
            print(f"train process {idx + 1} / {len(train_dataloader)}            ", end = "\r")
            data = data.to(args.device)
            labels = labels.to(args.device)
            
            output = model(data)
            loss = criterion(output, labels)
            
            _, preds = torch.max(output, 1)
            correct = torch.sum(preds == labels.data).cpu().numpy()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_correct += correct
            train_loss += loss.data.cpu() * len(data)
            train_data_num += len(data)

        model.eval()
        with torch.no_grad():
            for idx, (data, labels) in enumerate(val_dataloader):
                print(f"val process {idx + 1} / {len(val_dataloader)}             ", end = "\r")
                data = data.to(args.device)
                labels = labels.to(args.device)

                output = model(data)
                loss = criterion(output, labels)
                _, preds = torch.max(output, 1)
                correct = torch.sum(preds == labels.data).cpu().numpy()

                val_correct += correct
                val_loss += loss.data.cpu() * len(data)
                val_data_num += len(data)
            
        
        train_acc = train_correct / train_data_num
        train_loss = train_loss / train_data_num
        val_acc =  val_correct / val_data_num
        val_loss = val_loss / val_data_num
        print(f"[Epoch {epoch + 1}] Train acc {train_acc:.4f} loss {train_loss:.4f} || Val acc {val_acc:.4f} loss {val_loss:.4f}")
        
        if val_acc > best_val_acc:
            print(f"Save best at {epoch+1} {val_acc:.4f}")
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"./models/{args.model_name}.pth")
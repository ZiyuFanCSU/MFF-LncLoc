import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, recall_score
from configs import OUTPUTS_DIR, MODEL_PATH, DEVICE, CONFIG
from evaluation import Evaluation
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report


def softmax(x):
    # 计算每行的最大值
    row_max = np.max(x)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp)
    s = x_exp / x_sum
    return s

def train_epoch(train_loader, model, optimizer,scheduler, criterion, epoch):

    model.train()
    y_true = []
    y_pred = []
    losses = []
    for idx, batch_data in enumerate(train_loader):
        inputs0, inputs1, inputs2, features, targets = batch_data
        outputs = model((inputs0.to(DEVICE),inputs1.to(DEVICE),inputs2.to(DEVICE)), features.to(DEVICE))
        
        loss = criterion(outputs, targets.reshape(-1).to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_true+=[tensor.item() for tensor in targets.reshape(-1)]
        y_pred+=outputs.cpu().detach().numpy().tolist()
        losses.append(loss.item())
    scheduler.step()
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    accuracy = accuracy_score(y_true, torch.argmax(torch.tensor(y_pred), dim=1).cpu())
    probabilities = torch.softmax(torch.tensor(y_pred), dim=1).cpu().numpy()
    auc = roc_auc_score(y_true, probabilities, multi_class='ovr', average='micro',labels=[0, 1, 2, 3])
    aupr = average_precision_score(y_true, probabilities, average='micro')
    f1 = f1_score(y_true, torch.argmax(torch.tensor(y_pred), dim=1).cpu(), average='macro')
    recall = recall_score(y_true, torch.argmax(torch.tensor(y_pred), dim=1).cpu(), average='macro')
    precision = precision_score(y_true, torch.argmax(torch.tensor(y_pred), dim=1).cpu(), average='macro')
    print(f"[train] Epoch: {epoch} / {CONFIG['epoch']}, loss: {sum(losses)/len(losses)}, lr: {current_lr}, acc: {accuracy}, auc: {auc}, aupr: {aupr}, f1: {f1}, recall: {recall}, precision: {precision}")
    return accuracy, auc


def evaluate(val_loader, model, epoch):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []

        for idx, batch_data in enumerate(val_loader):
            inputs0, inputs1, inputs2, features, targets = batch_data
            outputs = model((inputs0.to(DEVICE),inputs1.to(DEVICE),inputs2.to(DEVICE)), features.to(DEVICE))
            y_true+=[tensor.item() for tensor in targets.reshape(-1)]
            y_pred+=outputs.cpu().detach().numpy().tolist()

        test_acc = accuracy_score(y_true, torch.argmax(torch.tensor(y_pred), dim=1).cpu())
        test_probabilities = torch.softmax(torch.tensor(y_pred), dim=1).cpu().numpy()
        test_auc = roc_auc_score(y_true, test_probabilities, multi_class='ovr',average="micro")
        test_auc2 = roc_auc_score(y_true, test_probabilities, multi_class='ovr',average="macro")
        test_aupr = average_precision_score(y_true, test_probabilities, average='micro')
        test_aupr2 = average_precision_score(y_true, test_probabilities, average='macro')
        test_f1 = f1_score(y_true, torch.argmax(torch.tensor(y_pred), dim=1).cpu(), average='macro')
        test_recall = recall_score(y_true, torch.argmax(torch.tensor(y_pred), dim=1).cpu(), average='macro')
        test_precision = precision_score(y_true, torch.argmax(torch.tensor(y_pred), dim=1).cpu(), average='macro')
        print(f"[valid] Epoch: {epoch}, acc: {test_acc}, auc: {test_auc}, aupr: {test_aupr}, f1: {test_f1}, recall: {test_recall}, precision: {test_precision}, auc2: {test_auc2}, aupr2: {test_aupr2}")
        
        num_classes = outputs.size(1)
        y_true_array = np.array(y_true)
        y_true_tensor = torch.tensor(y_true)
        y_pred_tensor = torch.tensor(y_pred)
        y_pred_labels = torch.argmax(y_pred_tensor, dim=1)
        y_true_binary = label_binarize(y_true_array, classes=range(num_classes))

        t = classification_report(y_true, y_pred_labels, target_names=[0,1,2,3],output_dict=True)

        each_test_acc = []
        each_test_precision = []
        each_test_recall = []
        each_test_f1 = []
        each_test_auc = []
        each_test_aupr = []

        for class_index in range(outputs.size(1)):  # 假设outputs是(batch_size, num_classes)
            class_mask = (y_true_tensor == class_index)
            class_true = y_true_tensor[class_mask]
            class_pred = y_pred_labels[class_mask]
            class_acc = accuracy_score(class_true, class_pred)
            class_recall = t[class_index]['recall']
            class_precision = t[class_index]['precision']
            class_f1 = t[class_index]['f1-score']
            class_scores = test_probabilities[:, class_index]
            class_auc = roc_auc_score(y_true_binary[:, class_index], class_scores)
            class_aupr = average_precision_score(y_true_binary[:, class_index], class_scores)
            each_test_acc.append(class_acc)
            each_test_precision.append(class_recall)
            each_test_recall.append(class_precision)
            each_test_f1.append(class_f1)
            each_test_auc.append(class_auc)
            each_test_aupr.append(class_aupr)

        
        return test_acc, test_precision, test_recall, test_f1, test_auc, test_aupr, test_auc2, test_aupr2


def train(train_loader, val_loader,test_loader, model, optimizer,scheduler, criterion, bestforwhat):

    best_f1 = 0
    best_valid_record = []
    best_test_record = []

    for epoch in range(1, CONFIG["epoch"] + 1):
        _, _ = train_epoch(train_loader, model, optimizer,scheduler, criterion, epoch)
        valid_acc, valid_pre, valid_recall, valid_fscore, valid_auc, valid_aupr,valid_auc2, valid_aupr2 = evaluate(val_loader, model, epoch)
        test_acc, test_pre, test_recall, test_fscore, test_auc, test_aupr,test_auc2, test_aupr2 = evaluate(test_loader, model, epoch)
        
        if bestforwhat == "f1":
            if best_f1 < valid_fscore:
                best_f1 = valid_fscore
                print(f'最高的F1 score为{best_f1}')
                best_valid_record = []
                best_valid_record.append(valid_acc)
                best_valid_record.append(valid_pre)
                best_valid_record.append(valid_recall)
                best_valid_record.append(valid_fscore)
                best_valid_record.append(valid_auc)
                best_valid_record.append(valid_aupr)
                best_valid_record.append(valid_auc2)
                best_valid_record.append(valid_aupr2)

                best_test_record = []
                best_test_record.append(test_acc)
                best_test_record.append(test_pre)
                best_test_record.append(test_recall)
                best_test_record.append(test_fscore)
                best_test_record.append(test_auc)
                best_test_record.append(test_aupr)
                best_test_record.append(test_auc2)
                best_test_record.append(test_aupr2)


    return best_valid_record, best_test_record

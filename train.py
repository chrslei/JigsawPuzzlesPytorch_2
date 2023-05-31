import argparse
import datetime

import time


import torch
import os
import numpy as np
import random
import pandas as pd
import shutil
import torch.nn as nn
from Dataset import FoldDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
#from JigsawNet import JigsawNet
#from JigsawNetAlexNet import JigsawNet
#from JigsawNetAlexNetPretrained import JigsawNet
#from JigsawNetVGG16Pretrained import JigsawNet
from JigsawNetResnet101Pretrained import JigsawNet
from tqdm import tqdm


# Funktion zur Initialisierung der Gewichte des Modells
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# Funktion zum Speichern der Modell-Checkpoints
def save_checkpoint(net, path, global_step, accuracy=None, info=''):
    try:
        os.makedirs(path)
        print('Created checkpoint directory')
    except OSError:
        pass
    if accuracy:
        checkpoint_name = f'CP_%d_%.4f%s.pth' % (global_step, accuracy, info)
    else:
        checkpoint_name = f'CP_{global_step}{info}.pth'
    torch.save(net.state_dict(),
               os.path.join(path, checkpoint_name))
    print(f'Checkpoint {checkpoint_name} saved !')
    print('')
    print('Starting Test...')
    print('')


# Funktion zum Sammeln der Kommandozeilenargumente
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16, dest='batch_size')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, dest='lr')
    parser.add_argument('-n', '--exp_name', type=str, default='exp', dest='exp_name')
    parser.add_argument('-e', '--epochs', type=int, default=50, dest='epochs')
    parser.add_argument('-s', '--seed', type=int, default=1, dest='seed')
    parser.add_argument('-m', '--model', type=str, default='ResNet101', dest='model')
    return parser.parse_args()


# Funktion zum Bewerten des Modells auf dem Validierungsdatensatz
def evaluate(model, val_loader, device):
    model.eval()
    all = 0
    p = 0
    for batch in val_loader:
        _, clips, labels = batch
        clips = clips.to(device)
        labels = labels.to(device, dtype=torch.long).squeeze()  # B * 1
        # ---- forward ----
        pred = model(clips)  # B * 100
        pred_label = torch.argmax(torch.softmax(pred, dim=1), dim=1).long()
        p += (pred_label == labels).sum().item()
        all += labels.size(0)
    model.train()
    return p / all


# Hauptfunktion zum Training des Modells
def train(train_loader, val_loader, model, optimizer, epochs, device, writer):
    # ----prepare ----
    model.to(device)
    model.train()
    total_step = 0
    criterion = nn.CrossEntropyLoss()
    # ---- training ----
    for epoch in range(1, epochs + 1):
        with tqdm(total=len(train_loader), desc=f'epoch[{epoch}/{epochs + 1}]:') as bar:
            for batch in train_loader:
                # ---- data prepare ----
                _, clips, labels = batch
                clips = clips.to(device)
                labels = labels.to(device, dtype=torch.long)
                # ---- forward ----
                preds = model(clips)  # B * 100
                # ---- loss ----
                loss = criterion(preds, labels)
                # ---- backward ----
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_step += 1
                # ---- log ----
                writer.add_scalar('info/loss', loss, total_step)
                bar.set_postfix(**{'loss (batch)': loss.item()})
                bar.update(1)
            # ---- validation ----
            accuracy = evaluate(model, val_loader, device)
            writer.add_scalar('eval/ac', accuracy, total_step)
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], total_step)
            print(f"""
                Accuracy: {accuracy * 100:.2f}%
                {int(accuracy * len(val_set))} / {len(val_set)} correct
            """)

    print('training finished')

    return accuracy


# Hauptteil des Skripts
if __name__ == '__main__':
    # ---- init ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    print("using device: ", device)
    args = get_args()

    # Bestimmen der Pfade zu den Datenverzeichnissen
    train_dir = 'ILSVRC2012_img_train_t3_split/train'
    val_dir = 'ILSVRC2012_img_train_t3_split/val'
    test_dir = 'ILSVRC2012_img_train_t3_split/test'

    # Bestimmen des Pfades für die Logs
    log_path = 'log/' + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + f"_{args.epochs}epoch_{args.batch_size}batch_{args.model}"

    # Erstellen des Log-Verzeichnisses, falls es noch nicht existiert
    try:
        os.makedirs(log_path)
    except:
        pass
    # ---- random seed ----
    # Setzen des Zufallsseeds für die Reproduzierbarkeit
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    # Setzen des Zufallsseeds für die Arbeiter des Dataloader
    def _init_fn(worker_id):
        np.random.seed(int(seed))


    # ---- log & dataset ----
    # Erstellen der Verzeichnisse für das Logging und die Datensätze
    if not os.path.exists(os.path.join(log_path, args.exp_name)):
        os.makedirs(os.path.join(log_path, args.exp_name))
    if os.path.exists(os.path.join(log_path, args.exp_name, 'log')):
        shutil.rmtree(os.path.join(log_path, args.exp_name, 'log'))
    writer = SummaryWriter(os.path.join(log_path, args.exp_name, 'log'))

    # Manuell hinzugefügt
    # Erzeugen der Dateilisten für die Trainings-, Validierungs- und Testdaten
    train_pool = os.listdir(train_dir)
    val_pool = os.listdir(val_dir)
    test_pool = os.listdir(test_dir)

    # Laden der Permutationen für das Jigsaw-Training
    permutations = np.load('permutations.npy').tolist()
    # Erzeugen der Datensätze für das Training, die Validierung und das Testen
    # Bei ResNet101 ist in_channels = 3, ansonsten 1
    train_set = FoldDataset(train_dir, train_pool, permutations, in_channels=3)
    # Bei ResNet101 ist in_channels = 3, ansonsten 1
    val_set = FoldDataset(val_dir, val_pool, permutations, in_channels=3)
    # Bei ResNet101 ist in_channels = 3, ansonsten 1
    test_set = FoldDataset(test_dir, test_pool, permutations, in_channels=3)
    # Erzeugen der Dataloader für das Training, die Validierung und das Testen
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=10, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=10, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=10, pin_memory=True)
    # ---- model ----

    # VGG 16 & AlexNet
    # model = JigsawNet(1, 100)

    # AlexNetPretrained
    # model = JigsawNet(100)

    # Erstellen des Modells
    # ResNet
    model = JigsawNet(3, 100)

    # Initialisieren des Modells mit bestimmten Gewichten
    model.apply(weight_init)
    # Erstellen des Optimizers
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    # Festlegen der Anzahl der Epochen
    epochs = args.epochs
    # Beginn des Trainings
    print(f'''
            training start! 
            train set num: {len(train_set)} 
            val set num: {len(val_set)}
            test set num: {len(test_set)}

            ''')
    ac = train(train_loader, val_loader, model, optimizer, epochs, device, writer)
    # Speichern des Modells nach dem Training
    save_checkpoint(model, os.path.join(log_path, args.exp_name, 'checkpoints'), 0, ac)

    ### TESTING ###
    # Auswertung des Modells auf den Testdaten
    test_accuracy = evaluate(model, test_loader, device)
    print(f"""Test Accuracy: {test_accuracy * 100:.2f}%""")
    print(f"""{int(test_accuracy * len(test_set))} / {len(test_set)} correct""")

    # Protokollierung der Testgenauigkeit
    writer.add_scalar('test/ac', test_accuracy)
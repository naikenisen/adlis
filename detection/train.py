from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    NUM_WORKERS, IMAGES_DIR
)
from model import create_model
from custom_utils import (
    Averager, SaveBestModel, save_loss_plot,
    evaluate_detection_model
)
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, create_valid_dataset,
    create_train_loader, create_valid_loader
)
from torch.optim.lr_scheduler import StepLR
import torch
import matplotlib.pyplot as plt
import time
import os

plt.style.use('ggplot')
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

def train_one_epoch(train_loader, model, optimizer, loss_hist):
    """
    Entraîne le modèle sur le train_loader, calcule la 'train_loss' moyenne.
    """
    print('Training...')
    model.train()
    prog_bar = tqdm(train_loader, total=len(train_loader))
    
    for images, targets in prog_bar:
        optimizer.zero_grad()
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)  # dict of RPN + ROI losses
        losses = sum(loss_dict.values())
        loss_value = losses.item()
        
        loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    return loss_hist.value  # moyenne sur l'epoch

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1) Création Dataset + DataLoader
    train_dataset = create_train_dataset(IMAGES_DIR)
    valid_dataset = create_valid_dataset(IMAGES_DIR)
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # 2) Créer le modèle (une classe + BG => num_classes=2)
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    
    # 3) Info sur le nb de params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    # 4) Optimiseur + scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    scheduler = StepLR(optimizer=optimizer, step_size=15, gamma=0.1, verbose=True)

    # 5) Objets pour suivre la loss
    train_loss_hist = Averager()
    train_loss_list = []

    # 6) SaveBestModel (sur F1, par exemple)
    save_best_model = SaveBestModel(best_metric=0.0)

    # 7) Boucle d'entraînement
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        start_time = time.time()
        
        train_loss_hist.reset()
        train_loss = train_one_epoch(train_loader, model, optimizer, train_loss_hist)

        # (A) On peut calculer P, R, F1 sur le valid_loader
        precision, recall, f1 = evaluate_detection_model(
            model, valid_loader, DEVICE,
            iou_threshold=0.5, score_threshold=0.5
        )

        end_time = time.time()
        duration = (end_time - start_time) / 60
        
        # (B) Logs
        print(f"Epoch #{epoch+1} train loss: {train_loss:.3f}")
        print(f"Epoch #{epoch+1} -> P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        print(f"Took {duration:.3f} minutes for epoch {epoch+1}")
        
        # (C) On stocke la loss pour tracer
        train_loss_list.append(train_loss)
        save_loss_plot(OUT_DIR, train_loss_list)

        # (D) On peut sauvegarder le meilleur modèle selon la F1
        save_best_model(model, f1, epoch, OUT_DIR)

        # (E) Step Scheduler
        scheduler.step()

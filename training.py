from statistics import mode
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
from .to_gpu import get_default_device
from .evaluation import mIoU, dice_coefficient
device=get_default_device()


#record epoch, lr, val_loss(mean of all batch losses), train_loss(mean of all train losses),
def train(epochs,max_lr,model,train_loader,val_loader,weight_decay=0,grad_clip=None,opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history=[]
    optimizer=opt_func(model.parameters(),max_lr,weight_decay=weight_decay)
    sched=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr,epochs=epochs,steps_per_epoch=len(train_loader))
    max_c=0
    for epoch in range(epochs):
        model.train()
        train_losses=[]
        val_losses=[]
        train_iou=[]
        val_iou=[]
        train_dice_coefficient=[]
        val_dice_coefficient=[]

        for images,masks in train_loader:
            images=images.to(device)
            masks=masks.to(device)
            out=model(images)  
            loss=F.cross_entropy(out,masks)
            loss.backward()
            train_losses.append(loss.item())
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(),grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            sched.step()
            z=mIoU(out,masks,n_classes=3).item()
            train_iou.append(z)
            d=dice_coefficient(out,masks)
            train_dice_coefficient.append(d)
            
            
            
        with torch.no_grad():
            for images,masks in val_loader:
                images=images.to(device)
                masks=masks.to(device)
                out=model(images)
                loss=F.cross_entropy(out,masks).item()
                val_losses.append(loss)
                
                x=mIoU(out, masks, n_classes=3).item()
                val_iou.append(x)
                y=dice_coefficient(out,masks)
                val_dice_coefficient.append(y)
            

        result=dict()
        result['epoch']=epoch+1
        result['train_loss']=np.mean(train_losses).item()
        result['val_loss']=np.mean(val_losses).item()
        result['train_iou']=np.mean(train_iou).item()
        result['val_iou']=np.mean(val_iou).item()
        result['train_dice_coefficient']=np.mean(train_dice_coefficient).item()
        result['val_dice_coefficient']=np.mean(val_dice_coefficient).item()

        print('epoch',epoch)
        print('train_loss',result['train_loss'])
        print('val_loss',result['val_loss'])
        print('train_iou',result['train_iou'])
        print('val_iou',result['val_iou'])
        print('train_dice_coefficient',result['train_dice_coefficient'])
        print('val_dice_coefficient',result['val_dice_coefficient'])
        history.append(result)
        

        #save model
        criteria=(result['val_dice_coefficient']+result['val_iou'])/2
        if max_c< criteria :
            max_c=criteria
            torch.save(model,'./ce500.pt')
            print('model saved! Criteria=',criteria)
    return history

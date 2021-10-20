import os
from pathlib import Path
import pytorch_lightning as pl
import torch
from donkeycar.parts.pytorch.my_torch_data import TorchTubDataModule
#from donkeycar.parts.pytorch.torch_data import TorchTubDataModule

from donkeycar.parts.pytorch.torch_utils import get_model_by_type
from torch.nn import MSELoss
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import timeit
import torchvision.models as models
import torch.nn as nn
import numpy as np

USE_CUDA = True
CUDA = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
if CUDA:
    print('run on %s' % device)
    torch.cuda.empty_cache()



def train(cfg, tub_paths, model_output_path, model_type):
    """
    Train the model
    """
    # model_name, model_ext = os.path.splitext(model_output_path)

    # is_torch_model = model_ext == '.ckpt'
    # if is_torch_model:
    #     model = f'{model_name}.ckpt'
    # else:
    #     print("Unrecognized model file extension for model_output_path: '{}'. Please use the '.ckpt' extension.".format(
    #         model_output_path))


    if not model_type:
        model_type = cfg.DEFAULT_MODEL_TYPE

    tubs = tub_paths.split(',')
    model_name =  model_output_path.split("\\")[-1].split(".")[0]
    tub_paths = [os.path.expanduser(tub) for tub in tubs]
    # output_path = os.path.expanduser(model_output_path)
    output_path = Path(model_output_path).parent
    
    print("Model name: ",model_name)
    print("Output path: ",output_path)
    if torch.cuda.is_available():
        print('Using CUDA')
    else:
        print('Not using CUDA')

    # logger = None
    # if cfg.VERBOSE_TRAIN:
    #     print("Tensorboard logging started. Run `tensorboard --logdir ./tb_logs` in a new terminal")
    #     from pytorch_lightning.loggers import TensorBoardLogger
    #     # Create Tensorboard logger
    #     logger = TensorBoardLogger('tb_logs', name=model_name)

    #model = get_model_by_type(model_type, cfg, checkpoint_path=checkpoint_path)
    data_module = TorchTubDataModule(cfg, tub_paths)
    train_data = data_module.train_dataloader()
    valid_data = data_module.val_dataloader()

    print("TRAIN len: ",len(train_data))
    print("VALID len: ",len(valid_data))
    trainiter = iter(train_data)
    images, labels = next(trainiter)
    print(images.shape, labels.shape)

    model = load_VGG('vgg16',2)
    #print(model)
    model.to(device)
    lr = 1e-3
    criterion = MSELoss()
    #criterion_multioutput = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, threshold=1e-5) 

    num_epochs=cfg.MAX_EPOCHS

    training(num_epochs, model=model, loss_func=criterion, opt=optimizer, scheduler=scheduler, train_dl=train_data , val_dl=valid_data, model_name=model_name, output_path=output_path)

    print('DONE')

def training(epochs, model, loss_func, opt, scheduler, train_dl, val_dl, model_name, output_path):
    
    stat_train = pd.DataFrame(columns=['epoch',  'train loss', 'train loss batch'])
    stat_valid = pd.DataFrame(columns=['epoch',  'valid loss', 'valid loss batch'])

    start_time = timeit.default_timer()
    #model.to(device)
    loss_func.to(device)
    for epoch in range(epochs):        
        model.train()      
        # print("sono qui")
        train_loss, train_loss_batch = loss_epoch(model,loss_func,train_dl,opt)
        # print("fine loss")
        print('TRAIN - Epoca[{}/{}] - train loss: {:.6f} ({:.6f})'.format(epoch+1, epochs,train_loss, np.array(train_loss_batch).mean()))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time))
        stat_train = stat_train.append({'epoch':epoch, 'train loss':train_loss, 'train loss batch':train_loss_batch}, ignore_index=True)

        # validation
        val_loss, valid_loss_batch = validation(epoch, model, loss_func, val_dl,output_path,  model_name)
        stat_valid = stat_valid.append({'epoch':epoch, 'valid loss':val_loss, 'valid loss batch':valid_loss_batch}, ignore_index=True)
       
        # aggiusto learning rate
        scheduler.step(val_loss)
        
        # Empty cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        #if(epoch%1==0 and epoch>0):
            # salvo modello e parametri
        save_model(epoch, model, opt, loss_func, scheduler, stat_train,stat_valid, model_name, output_path)
        
    return model

def loss_epoch(model, loss_func, dataset_dl, opt=None):    
    lossTot = 0.0   
    loss_batch = []
    
    for inputs, labels in dataset_dl: 
        # data to cuda
        inputs = Variable(inputs.to(device), requires_grad=True)       
        labels = Variable(labels.to(device), requires_grad=False) 
        # opt.zero_grad()   
        # obtain model output        
        predict = model(inputs)
        # print(predict)
        # obtain loss    
        loss = loss_func(predict, labels) 
        loss_b = loss.item()
        # print(loss_b)
        # obtain performance accuracy
        if opt is not None: # only training
            loss.backward()        
            opt.step()        
            opt.zero_grad()
        # print("fine loss")
        # stat
        loss_batch.append(loss_b)
        lossTot+=loss_b
    
    lossTot /= len(dataset_dl)     
    
    return lossTot, loss_batch

def save_model(epoch, model, opt, loss_func, scheduler, stat_train, stat_valid, name_model, output_path):
    if not os.path.isdir(output_path):
            os.mkdir(output_path)
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'opt_dict': opt.state_dict(),
        'loss_dict': loss_func.state_dict(),
        'sheduler_dict': scheduler.state_dict()
    }, os.path.join(output_path, name_model+'_epoch-' + str(epoch) + '.pth'))
    
    stat_train.to_csv(os.path.join(output_path, name_model+'_train_epoch-' + str(epoch) + '.csv'))
    stat_valid.to_csv(os.path.join(output_path, name_model+'_valid_epoch-' + str(epoch) + '.csv'))

    print("Save model at {}".format(os.path.join(output_path, name_model+'_epoch-' + str(epoch) + '.pth')))
        
def validation(epoch, model, loss_func, val_dl, output_path, model_name):
    #stat = pd.DataFrame(columns=['epoch',  'valid loss', 'loss batch'])
    start_time = timeit.default_timer()
    #model.to(device)
    #loss_func.to(device)
    model.eval()        
    
    with torch.no_grad():            
        val_loss,  loss_batch = loss_epoch(model, loss_func, val_dl)        

    print("VALIDATION - Epoca {} - val loss: {:.6f} ({:.6f})".format(epoch, val_loss, np.array(loss_batch).mean()) )
        
    #stat = stat.append({'epoch':epoch, 'valid loss':val_loss, 'loss batch':loss_batch}, ignore_index=True)
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")
        
    # stat.to_csv(os.path.join(output_path, model_name+'_valid_epoch-' + str(epoch) + '.csv'))
    return val_loss, loss_batch

# def accuracys_batch(target, output):    
#     # obtain output class    
#     pred = output.argmax(dim=1, keepdim=True)    
#     # compare output class with target class    
#     corrects = pred.eq(target.view_as(pred)).sum().item()    
#     return corrects

def load_VGG(type_vgg, num_classes=2):
    # Load the pre-trained model (on ImageNet)
    model = None
    if type_vgg == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("-my_torch_train NO MODEL selectd")

    # Don't allow model feature extraction layers to be modified
    for layer in model.features.parameters():
        layer.requires_grad = False

    # Change the classifier layer
    model.classifier._modules['6'] = nn.Linear(4096, num_classes)

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
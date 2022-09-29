import sys
sys.path.append("..")
import os 
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from time import time
from UCI_HAR_load import UCI_HAR_load, UCI_HAR_Dataset
from model import TStransformer
from Early_Stopping import EarlyStopping
from learning_rate import adjust_learning_rate_class



class Exp_TStransformer(object):
    def __init__(self, args):
        self.args = args

        self.device = self._acquire_device()

        # 数据集
        self.train_data_set, self.train_data_loader, self.vali_data_set, self.vali_data_loader, self.test_data_set, self.test_data_loader = self._get_data()
        self.input_dimension = self.train_data_set.data_channel
        
        
        self.model = self._build_model().to(self.device)
        
        self.optimizer_dict = {"Adam":optim.Adam}
        self.criterion_dict = {"CrossEntropy":nn.CrossEntropyLoss}



    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:0')
            print('Use GPU: cuda:0')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self):
        model = TStransformer(enc_in              = self.input_dimension,
                            input_length          = self.args.sequence_length,
                            c_out                 = self.args.class_num,
                            d_model               = self.args.d_model,
                            attention_layer_types = self.args.attention_layer_types,
                            embedd_kernel_size    = self.args.embedd_kernel_size,
                            forward_kernel_size   = self.args.forward_kernel_size,
                            value_kernel_size     = self.args.value_kernel_size,
                            causal_kernel_size    = self.args.causal_kernel_size,
                            d_ff                  = self.args.d_ff,
                            n_heads               = self.args.n_heads,
                            e_layers              = self.args.e_layers,
                            dropout               = self.args.dropout,
                            norm                  = self.args.norm,
                            activation            = self.args.activation,
                            output_attention      = self.args.output_attention,
                            predictor_type        = self.args.predictor_type)
        print("Parameter :", np.sum([para.numel() for para in model.parameters()]))
        return model.double()
    
    
    def _select_optimizer(self):
        # 这两个在train里面被调用
        if self.args.optimizer not in self.optimizer_dict.keys():
            raise NotImplementedError
            
        model_optim = self.optimizer_dict[self.args.optimizer](self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    
    def _select_criterion(self):
        # 这两个在train里面被调用
        if self.args.criterion not in self.criterion_dict.keys():
            raise NotImplementedError
            
        criterion = self.criterion_dict[self.args.criterion](reduction="mean")
        return criterion
		
    def _get_data(self, flag="train"):
        args = self.args
        X_train, X_vali, X_test, y_train, y_vali, y_test = UCI_HAR_load(args.PATH, args.split_rate)
        train_data_set = UCI_HAR_Dataset(X_train, y_train[:,0])
        vali_data_set = UCI_HAR_Dataset(X_vali, y_vali[:,0])
        test_data_set  = UCI_HAR_Dataset(X_test, y_test[:,0])
        
			
        train_data_loader = DataLoader(train_data_set, 
                                        batch_size  = args.batch_size,
                                        shuffle     = True,
                                        num_workers = 0,
                                        drop_last   = False)
        vali_data_loader  = DataLoader(vali_data_set, 
                                        batch_size  = args.batch_size,
                                        shuffle     = False,
                                        num_workers = 0,
                                        drop_last   = False)

            
        test_data_loader  = DataLoader(test_data_set, 
                                        batch_size  = args.batch_size,
                                        shuffle     = False,
                                        num_workers = 0,
                                        drop_last   = False)
        
        return train_data_set, train_data_loader, vali_data_set, vali_data_loader, test_data_set, test_data_loader

    
    
    def train(self, save_path):

        # 中间过程存储地址
        path = './logs/'+save_path
        if not os.path.exists(path):
            os.makedirs(path)
        
        # 根据batch ssize 看一个epoch里面有多少训练步骤 以及validation的步骤
        train_steps = len(self.train_data_loader)
        print("train_steps: ",train_steps)
        print("test_steps: ",len(self.vali_data_loader))
        
        # 初始化 早停止
        early_stopping = EarlyStopping(patience=self.args.early_stop_patience, verbose=True)
        # 初始化 学习率
        learning_rate_adapter = adjust_learning_rate_class(self.args, True)
        
        # 选择优化器
        model_optim = self._select_optimizer()
        
        # 选择优化的loss function
        loss_criterion =  self._select_criterion()
        

        print("start training")
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            start_time = time()
            self.model.train()
            

            for i, (batch_x, batch_y) in enumerate(self.train_data_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.long().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x)[0]
                else:
                    outputs = self.model(batch_x)
                # print(outputs.shape, "outputs")
                # print(batch_y.shape, "batch_y")
                loss = loss_criterion(outputs, batch_y)
                
                train_loss.append(loss.item())
                loss.backward()

            end_time = time()	
            epoch_time = end_time - start_time
            
            train_loss = np.average(train_loss) # 这个整个epoch中的平均loss
            
            vali_loss  = self.validation(self.vali_data_loader, loss_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}. it takse {4:.7f} seconds".format(
                epoch + 1, train_steps, train_loss, vali_loss, epoch_time))

            # acc, f_w,  f_macro, f_micro = self.test(self.test_data_loader)
            # print("test performace: | accuracy: {0:.7f}, f1_weight: {1:.7f} | f1_macro: {2:.7f}, f1_micro: {3:.7f}".format(acc, f_w,  f_macro, f_micro))

            # 在每个epoch 结束的时候 进行查看需要停止和调整学习率
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            learning_rate_adapter(model_optim,vali_loss)

        # test
        acc, f_w,  f_macro, f_micro = self.test(self.test_data_loader)
        print("test performace: | accuracy: {0:.7f}, f1_weight: {1:.7f} | f1_macro: {2:.7f}, f1_micro: {3:.7f}".format(acc, f_w,  f_macro, f_micro))

        last_model_path = path+'/'+'last_checkpoint.pth'
        torch.save(self.model.state_dict(), last_model_path)                
        

    def validation(self, data_loader, criterion):
        
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(data_loader):

                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.long().to(self.device)

                # prediction
                if self.args.output_attention:
                    outputs = self.model(batch_x)[0]
                else:
                    outputs = self.model(batch_x)

                pred = outputs.detach()#.cpu()
                true = batch_y.detach()#.cpu()

                loss = criterion(pred, true) 
                total_loss.append(loss.cpu())

                preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
                trues.extend(list(batch_y.detach().cpu().numpy()))   

        vali_loss = np.average(total_loss)
        # acc = accuracy_score(preds,trues)

        self.model.train()
        return vali_loss 


    def test(self, data_loader):    
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(data_loader):

                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.long().to(self.device)
              
                if self.args.output_attention:
                    outputs = self.model(batch_x)[0]
                else:
                    outputs = self.model(batch_x)

                pred = outputs.detach()#.cpu()
                true = batch_y.detach()#.cpu()

                preds.extend(list(np.argmax(outputs.detach().cpu().numpy(),axis=1)))
                trues.extend(list(batch_y.detach().cpu().numpy()))   

        acc = accuracy_score(preds,trues)

        f_w = f1_score(trues, preds, average='weighted')
        f_macro = f1_score(trues, preds, average='macro')
        f_micro = f1_score(trues, preds, average='micro')
        self.model.train()

        return acc, f_w, f_macro, f_micro   
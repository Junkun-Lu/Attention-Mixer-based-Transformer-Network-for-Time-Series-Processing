import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from Prodictor import FullPredictor, ConvPredictor, LinearPredictor
from Embedding import DataEmbedding
from Encoder import EncoderLayer, Encoder

class TStransformer(nn.Module):
    def __init__(self, 
          enc_in, 
          input_length, 
          c_out, 
          d_model=512, 
          attention_layer_types=["Triangular"],
          
          embedd_kernel_size = 3, 
          forward_kernel_size =3,
          value_kernel_size = 1,
          causal_kernel_size=3, 
          
          d_ff =None,
          n_heads=8, 
          e_layers=3,  
          dropout=0.1, 
          norm = "batch", 
          activation='relu', 
          output_attention = True,
  

          predictor_type = "linear"):
        
        """
        enc_in : 输入给encoder的channel数，也就是最开始的channel数, 这个通过dataloader获得
        input_length: 数据的原始长度，最后预测的时候要用，也要从dataloader获得
        c_out ： 最后的输出层，这里应该等于原始输入长度
        d_model：每一层encoding的数量 ，这个数基本不变，因为在transofomer 中的相加 residual， d_model 就不变化
        attention_layer_types 一个list 包含那些attention的类型  ["Full", "Local", "LocalLog", "ProbMask"]     
        n_heads 总共attention多少个头，目前大概是三的倍数
        e_layers： 多少层encoder
        
        
        """
        super(TStransformer, self).__init__()

        self.enc_in                 = enc_in
        self.d_model                = d_model
        self.embedd_kernel_size     = embedd_kernel_size
        self.dropout                = dropout
        
        self.attention_layer_types = attention_layer_types
        self.n_heads               = n_heads
        self.d_ff                  = d_ff
        self.activation            = activation
        self.forward_kernel_size   = forward_kernel_size
        self.value_kernel_size     = value_kernel_size
        self.causal_kernel_size    = causal_kernel_size
        self.norm                  = norm
        self.output_attention      = output_attention
        self.e_layers              = e_layers
        
        self.input_length          = input_length
        self.c_out                 = c_out   # 有几个类

        self.predictor_type        = predictor_type
        # Encoding
        
        self.enc_embedding = DataEmbedding(c_in = enc_in, 
                                           d_model = d_model,
                                           embedd_kernel_size=embedd_kernel_size,
                                           dropout=dropout).double()

        
        
        # Encoder        
        self.encoder = Encoder([EncoderLayer(attention_layer_types = self.attention_layer_types,
                                             d_model               = self.d_model,
                                             n_heads               = self.n_heads,
                                             d_ff                  = self.d_ff,
                                             dropout               = self.dropout,
                                             activation            = self.activation,
                                             forward_kernel_size   = self.forward_kernel_size,
                                             value_kernel_size     = self.value_kernel_size,
                                             causal_kernel_size    = self.causal_kernel_size,
                                             output_attention      = self.output_attention,
                                             norm                  = self.norm) for l in range(self.e_layers)]
                               ).double()

        # 这里的输出是 （B， L, d_model） 

        if self.predictor_type == "full":
            self.predictor = FullPredictor(d_model, input_length).double()
        if self.predictor_type == "linear":
            self.predictor = LinearPredictor(d_model).double()
        if self.predictor_type == "conv":
            self.predictor = ConvPredictor(d_model = d_model, pred_kernel = 3).double()
        if self.predictor_type == "hybrid":
            self.predictor1 = FullPredictor(d_model, input_length).double()
            self.predictor2 = LinearPredictor(d_model).double()
            self.predictor3 = ConvPredictor(d_model = d_model, pred_kernel = 3).double()	
            self.predictor  = nn.Conv1d(in_channels = 3, out_channels = 1, kernel_size  = 3).double()	

        # 分类预测结果
        self.class_projection = nn.Linear(input_length, c_out).double()
        self.class_activ = nn.ReLU(inplace=True)
        self.class_dropout = nn.Dropout(dropout)

        # self.softmax = nn.Softmax(dim=1)
       
    def forward(self, x):
        # x shape 是 batch， L， Enc_in
        
        enc_out = self.enc_embedding(x)
        enc_out, attns = self.encoder(enc_out)

        if self.predictor_type == "hybrid":
            pred1 = self.predictor1(enc_out)
            #print(pred1.shape)
            if len(pred1.shape)==1:
                pred1 = torch.unsqueeze(pred1, 0)
            pred1 = torch.unsqueeze(pred1, 2)

            pred2 = self.predictor2(enc_out)
            #print(pred2.shape)
            if len(pred2.shape)==1:
                pred2 = torch.unsqueeze(pred2, 0)
            pred2 = torch.unsqueeze(pred2, 2)

            pred3 = self.predictor3(enc_out) 
            #print(pred3.shape)
            if len(pred3.shape)==1:
                pred3 = torch.unsqueeze(pred3, 0)
            pred3 = torch.unsqueeze(pred3, 2)

            hybrid_pred = torch.cat([pred1,pred2,pred3],dim=-1) 
            enc_out  = nn.functional.pad(hybrid_pred.permute(0, 2, 1), 
                                         pad=(1, 1),
                                         mode='replicate')
            enc_pred = self.predictor(enc_out).permute(0, 2, 1).squeeze()
        else:
            enc_pred = self.predictor(enc_out) # 这里的形状是 【B,L】

        if len(enc_pred.shape)==1:
            enc_pred = torch.unsqueeze(enc_pred, 0)

        enc_pred = self.class_dropout(self.class_activ(self.class_projection(enc_pred)))
        # enc_pred = self.softmax(enc_pred)
        

        if self.output_attention:
            return enc_pred, attns   # 在training时要注意这里是个元组, 所以out = self.model(batch_x)[0]
        else:
            return enc_pred # [B, L]
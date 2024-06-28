import torch
import torch.nn as nn
import snntorch as snn
import brevitas.nn as qnn 
from snntorch import utils
from snntorch import surrogate
import torch.nn.functional as F
from snntorch import functional as SF
import snntorch.functional.quant as quant
from RecurrentAHPC import RecurrentAhpc, QuantRecurrentAhpc
import numpy as np
class AhpcNetwork(nn.Module):
    def __init__(self, num_inputs, num_hidden_1, num_hidden_2, num_outputs, beta, back_beta, grad, alpha=None, device='cuda'):
        super(AhpcNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden_1)

        self.leaky1 = snn.Leaky(beta=beta, spike_grad=grad,
                                learn_beta=True, learn_threshold=True)
        
        self.linear2 = nn.Linear(num_hidden_1, num_hidden_2)

        self.recurrent = RecurrentAhpc(beta, back_beta, spike_grad=grad, back_grad=grad, linear_features = num_hidden_2,
                                       init_hidden=True, reset_delay=True, learn_beta=True,
                                       learn_threshold=True, learn_recurrent=True, alpha=alpha)
        
        self.linear3 = nn.Linear(num_hidden_2, num_outputs)
        self.leaky2 = snn.Leaky(beta=beta, spike_grad=grad,
                                learn_beta=True, learn_threshold=True, output=True)

    def forward(self, data):
        spk_rec = []
        utils.reset(self)  # resets hidden states for all LIF neurons in net
        
        for step in range(data.size(2)):
            x = self.linear1(data[:,:,step])
            x, _ = self.leaky1(x)
            x = self.linear2(x)
            spk1 = self.recurrent(x)
            x = self.linear3(spk1)
            x, _ = self.leaky2(x)
            spk_rec.append(x)
        batch_out = torch.stack(spk_rec)

        return batch_out
    
class QuantAhpcNetworkOLD(nn.Module):
    def __init__(self, num_inputs, num_hidden_1, num_hidden_2, 
                num_outputs, beta,  grad, threshold,
                alpha=None, back_beta=None, num_bits=8):
        
        super(QuantAhpcNetwork, self).__init__()

        if back_beta is None:
            back_beta = beta
        
        self.linear1 = qnn.QuantLinear(num_inputs, num_hidden_1, bias=False,
                                           weight_bit_width=num_bits)


        self.leaky1 = snn.Leaky(beta=beta, 
                                spike_grad=grad,
                                threshold= threshold, 
                                learn_threshold=True, 
                                learn_beta=True, 
                                reset_mechanism='zero',
                                reset_delay=False)
        
        self.linear2 = qnn.QuantLinear(num_hidden_1, num_hidden_2, bias=False,
                                        weight_bit_width=num_bits,
                                        weight_quant= self.linear1.weight_quant)

        self.recurrent = QuantRecurrentAhpc(beta, back_beta, spike_grad=grad, back_grad=grad, linear_features = num_hidden_2,
                                       init_hidden=True, reset_delay=True, learn_beta=True,
                                       learn_threshold=True, learn_recurrent=True, alpha=alpha,
                                       shared_weight_quant=  self.linear1.weight_quant)
        
        self.linear3 = qnn.QuantLinear(num_hidden_2, num_outputs, bias=False,
                                                        weight_bit_width=num_bits,
                                                        weight_quant=  self.linear1.weight_quant)
        
        self.leaky2 = snn.Leaky(beta=beta, 
                                spike_grad=grad,
                                threshold= threshold, 
                                learn_threshold=True, 
                                learn_beta=True, 
                                reset_mechanism='zero',
                                reset_delay=False,
                                output=True)

    def forward(self, data):
        spk_rec = []
        utils.reset(self)  # resets hidden states for all LIF neurons in net

        for step in range(data.size(2)):
            x = self.linear1(data[:,:,step])
            x, _ = self.leaky1(x)
            x = self.linear2(x)
            spk1 = self.recurrent(x)
            x = self.linear3(spk1)
            x, _ = self.leaky2(x)
            spk_rec.append(x)
        batch_out = torch.stack(spk_rec)

        return batch_out
    
class QuantAhpcNetwork(nn.Module):
    def __init__(self, num_inputs, num_hidden_1, num_hidden_2, 
                num_outputs, beta,  grad, threshold, time_dim=1,
                alpha=None, back_beta=None, num_bits=8, encoder_dim=None, layer_loss=None):
        super(QuantAhpcNetwork, self).__init__()

        self.layer_loss = layer_loss

        if back_beta is None:
            back_beta = beta

        self.time_dim = time_dim

        if encoder_dim is not None:
            self.encoder = True
            self.encoder_connection = qnn.QuantLinear(num_inputs, encoder_dim, bias=False,
                                            weight_bit_width=num_bits)


            self.encoder_population = snn.Leaky(beta=beta, 
                                    spike_grad=grad,
                                    threshold= threshold, 
                                    learn_threshold=True, 
                                    learn_beta=True, 
                                    reset_mechanism='zero',
                                    reset_delay=False)
            
            self.linear1 = qnn.QuantLinear(encoder_dim, num_hidden_1, bias=False,
                                           weight_bit_width=num_bits)
        else:
            self.encoder = False
            self.linear1 = qnn.QuantLinear(num_inputs, num_hidden_1, bias=False,
                                           weight_bit_width=num_bits)

           

        self.leaky1 = snn.Leaky(beta=beta, 
                                spike_grad=grad,
                                threshold= threshold, 
                                learn_threshold=True, 
                                learn_beta=True, 
                                reset_mechanism='zero',
                                reset_delay=False)
        
        self.linear2 = qnn.QuantLinear(num_hidden_1, num_hidden_2, bias=False,
                                        weight_bit_width=num_bits,
                                        weight_quant=  self.linear1.weight_quant)
        
        self.first_dropout = nn.Dropout1d(p=0.2)  

        self.recurrent = QuantRecurrentAhpc(beta, back_beta, spike_grad=grad, back_grad=grad, linear_features = num_hidden_2,
                                       init_hidden=True, reset_delay=True, learn_beta=True,
                                       learn_threshold=True, learn_recurrent=True, alpha=alpha,
                                       shared_weight_quant=  self.linear1.weight_quant)
        
        self.linear3 = qnn.QuantLinear(num_hidden_2, num_outputs, bias=False,
                                                        weight_bit_width=num_bits,
                                                        weight_quant=  self.linear1.weight_quant)
        self.second_dropout = nn.Dropout1d(p=0.1)

        self.leaky2 = snn.Leaky(beta=beta, 
                                spike_grad=grad,
                                threshold= threshold, 
                                learn_threshold=True, 
                                learn_beta=True, 
                                reset_mechanism='zero',
                                reset_delay=False,
                                output=True)

    def forward(self, data):
        spk_rec = []
        # utils.reset(self)  # resets hidden states for all LIF neurons in net
        if self.encoder:
            self.encoder_population.reset_hidden()
        self.leaky1.reset_hidden()
        self.recurrent.reset_hidden()
        self.leaky2.reset_hidden()

        dims = list(range(data.dim()))  # Creates a list of dimensions
        dims.pop(self.time_dim)                     # Remove the selected dimension
        dims.insert(0, self.time_dim)               # Insert the selected dimension at the front
        data_permuted = data.permute(dims)  # Permute the tensor
        if self.layer_loss is not None:
            if self.encoder:
                layer1_acc = []
                layer2_acc = []
                encoder_acc = []
            else:
                layer1_acc = []
                layer2_acc = []

        for slice in data_permuted:
            if self.encoder:
               
                x = self.encoder_connection(slice)
                x, _ = self.encoder_population(x)
                
                if self.training and self.layer_loss is not None:
                    encoder_acc.append(x.clone().detach().cpu())


                x = self.linear1(x)
                x, _ = self.leaky1(x)
                
                if self.training and self.layer_loss is not None:
                    layer1_acc.append(x.clone().detach().cpu())   

            else:
                x = self.linear1(slice)
                x, _ = self.leaky1(x)
                
                if self.layer_loss is not None:
                    layer1_acc.append(x.clone().detach().cpu())
                

            x = self.linear2(x)
            x = self.first_dropout(x)
            
            spk1 = self.recurrent(x)
            
            if self.training and self.layer_loss is not None:
                layer2_acc.append(spk1.clone().detach().cpu())

            x = self.linear3(spk1)
            x = self.second_dropout(x)
            
            x, _ = self.leaky2(x)
            
            spk_rec.append(x)
        
        net_loss = 0
        
        if self.training and self.layer_loss is not None:
            net_loss = self.layer_loss([torch.stack(layer1_acc), torch.stack(layer2_acc), torch.stack(encoder_acc)])

        batch_out = torch.stack(spk_rec)

        return batch_out, net_loss
    
    def save_to_npz(self, path):
        
        linear1 = self.linear1.weight.data.detach().cpu().numpy()
        leaky1_betas = self.leaky1.beta.data.detach().cpu().numpy()
        leaky1_vth = self.leaky1.threshold.data.detach().cpu().numpy()

        linear2 = self.linear2.weight.data.detach().cpu().numpy()

        recurrent_betas = self.recurrent.beta.data.detach().cpu().numpy()
        recurrent_vth = self.recurrent.threshold.data.detach().cpu().numpy()
        input_dense, activation, output_dense = self.recurrent.recurrent.to_npz()

        linear3 = self.linear3.weight.data.detach().cpu().numpy()
        leaky2_betas = self.leaky2.beta.data.detach().cpu().numpy()
        leaky2_vth = self.leaky2.threshold.data.detach().cpu().numpy()

        if self.encoder:
            encoder_connection = self.encoder_connection.weight.data.detach().cpu().numpy()
            encoder_population_betas = self.encoder_population.beta.data.detach().cpu().numpy()
            encoder_population_vth = self.encoder_population.threshold.data.detach().cpu().numpy()
            
            np.savez_compressed(path, encoder_connection=encoder_connection, encoder_population_betas=encoder_population_betas,
                                encoder_population_vth=encoder_population_vth, linear1=linear1, leaky1_betas=leaky1_betas,
                                leaky1_vth=leaky1_vth, linear2=linear2, recurrent_betas=recurrent_betas, recurrent_vth=recurrent_vth,
                                input_dense=input_dense, activation=activation, output_dense=output_dense,
                                linear3=linear3, leaky2_betas=leaky2_betas, leaky2_vth=leaky2_vth)
        else:
            np.savez_compressed(path, linear1=linear1, leaky1_betas=leaky1_betas,
                                leaky1_vth=leaky1_vth, linear2=linear2, recurrent_betas=recurrent_betas, recurrent_vth=recurrent_vth,
                                input_dense=input_dense, activation=activation, output_dense=output_dense,
                                linear3=linear3, leaky2_betas=leaky2_betas, leaky2_vth=leaky2_vth)
    def from_npz(self, path):
       
        data = np.load(path,allow_pickle=True)
        if self.encoder:
            self.encoder_connection.weight.data = torch.tensor(data['encoder_connection'])
            self.encoder_population.beta.data = torch.tensor(data['encoder_population_betas'])
            self.encoder_population.threshold.data = torch.tensor(data['encoder_population_vth'])
    
        self.linear1.weight.data = torch.tensor(data['linear1'])
        self.leaky1.beta.data = torch.tensor(data['leaky1_betas'])
        self.leaky1.threshold.data = torch.tensor(data['leaky1_vth'])

        self.linear2.weight.data = torch.tensor(data['linear2'])

        self.recurrent.beta.data = torch.tensor(data['recurrent_betas'])
        self.recurrent.threshold.data = torch.tensor(data['recurrent_vth'])

        self.recurrent.recurrent.from_npz(data['input_dense'], data['activation'], data['output_dense'])

        self.linear3.weight.data = torch.tensor(data['linear3'])
        self.leaky2.beta.data = torch.tensor(data['leaky2_betas'])
        self.leaky2.threshold.data = torch.tensor(data['leaky2_vth'])
    
    def print_params(self):
        if self.encoder :
            print("Encoder Population")
            print(self.encoder_population.beta)
            print(self.encoder_population.threshold)

        print("Leaky 1")
        print(self.leaky1.beta)
        print(self.leaky1.threshold)
        print("Recurrent")
        print(self.recurrent.beta)
        print(self.recurrent.threshold)
        print("Leaky 2")
        print(self.leaky2.beta)
        print(self.leaky2.threshold)
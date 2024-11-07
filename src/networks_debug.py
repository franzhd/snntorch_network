import torch
import torch.nn as nn
import snntorch as snn
from snntorch.functional import probe
import brevitas.nn as qnn 
from snntorch import utils
from snntorch import surrogate
import torch.nn.functional as F
from snntorch import functional as SF
from snntorch.functional import quant
from RecurrentAHPC_debug import QuantRecurrentAhpc
from brevitas.quant import Int8WeightPerTensorFixedPoint
import numpy as np

def state_quant_fn(input_, num_bits=8, threshold=1, lower_limit=0, upper_limit=0.2): # <-- VitF

    num_levels = 2 << num_bits - 1

    levels = torch.linspace(
                threshold - threshold * lower_limit,
                threshold + threshold * upper_limit,
                num_levels,
             )
    
    device = input_.device
    levels = levels.to(device)

    """
    # The original implementation

    size = input_.size()
    input_ = input_.flatten()
    # Broadcast mem along new direction same # of times as num_levels
    repeat_dims = torch.ones(len(input_.size())).tolist()
    repeat_dims.append(levels.shape[0]) # .append(len(levels))
    repeat_dims = [int(item) for item in repeat_dims]
    repeat_dims = tuple(repeat_dims)
    input_ = input_.unsqueeze(-1).repeat(repeat_dims)
    # find closest valid quant state
    idx_match = torch.min(torch.abs(levels - input_), dim=-1)[1]
    quant_tensor = levels[idx_match]
    
    return quant_tensor.reshape(size)
    """

    differences = torch.abs(input_.unsqueeze(-1) - levels)
    closest_indices = torch.argmin(differences, dim=-1)
    
    return levels[closest_indices]
    
class QuantAhpcNetwork(nn.Module):
    def __init__(self,
                num_inputs, num_hidden_1, num_hidden_2, num_outputs,
                grad,
                vth_in, vth_recurrent, vth_back, vth_out, 
                beta_in, beta_recurrent, beta_back, beta_out,
                encoder_dim=None, vth_enc_value=1.0, vth_std=1000,beta_std=1000,
                drop_recurrent=0.0, drop_back=0.0, drop_out=0.0,
                state_quant=False,
                time_dim=1,
                num_bits=8,
                layer_loss=None,
                weight_quant=Int8WeightPerTensorFixedPoint):
        
        super(QuantAhpcNetwork, self).__init__()
        self.layer_loss = layer_loss

        self.time_dim = time_dim
        
        self.quant = state_quant

        if encoder_dim is not None:
            self.encoder = True
            vth_e = self.gen_gaussian_distribution(encoder_dim, encoder_dim/2, vth_std, vth_enc_value)
            beta_e = self.gen_gaussian_distribution(encoder_dim, encoder_dim/2, beta_std)
            
            self.encoder_connection = qnn.QuantLinear(num_inputs, encoder_dim, bias=False,
                                            weight_bit_width=num_bits)
            #self.encoder_connection = nn.Linear(num_inputs, encoder_dim, bias=False)
           
            self.encoder_population = snn.Leaky(beta=beta_e, 
                                    spike_grad=grad,
                                    threshold= vth_e, 
                                    learn_threshold=True, 
                                    learn_beta=True, 
                                    reset_mechanism='zero',
                                    reset_delay=False)
            
            self.linear1 = qnn.QuantLinear(encoder_dim, num_hidden_1, bias=False,
                                           weight_bit_width=num_bits)
        else:
            self.encoder = False
            self.linear1 = qnn.QuantLinear(num_inputs, num_hidden_1, bias=False,
                                           weight_bit_width=num_bits, weight_quant=weight_quant) 

           

        self.leaky1 = snn.Leaky(beta=beta_in, 
                                spike_grad=grad,
                                threshold= vth_in, 
                                learn_threshold=True, 
                                learn_beta=True, 
                                reset_mechanism='zero',
                                reset_delay=False,
                                state_quant=self.quant)

        self.linear2 = qnn.QuantLinear(num_hidden_1, num_hidden_2, bias=False,
                                        weight_bit_width=num_bits,
                                        weight_quant=  self.linear1.weight_quant)
        
        self.dropout_rec = nn.Dropout(p=drop_recurrent)  

        self.recurrent = QuantRecurrentAhpc(beta_recurrent, beta_back, vth_back, spike_grad=grad, linear_features = num_hidden_2,
                                            init_hidden=False, reset_delay=False, learn_beta=True,
                                            learn_threshold=True, learn_recurrent=True, 
                                            threshold=vth_recurrent, reset_mechanism="zero",
                                            shared_weight_quant=self.linear1.weight_quant,state_quant=self.quant, dropout=drop_back, output=True)
        self.linear3 = qnn.QuantLinear(num_hidden_2, num_outputs, bias=False,
                                                        weight_bit_width=num_bits,
                                                        weight_quant=  self.linear1.weight_quant)
        self.dropout_out = nn.Dropout(p=drop_out)

        self.leaky2 = snn.Leaky(beta=beta_out, 
                                spike_grad=grad,
                                threshold= vth_out, 
                                learn_threshold=True, 
                                learn_beta=True, 
                                reset_mechanism='zero',
                                reset_delay=False,
                                output=True,
                                state_quant=self.quant)
    def forward(self, data):
        spk_rec = []
        # utils.reset(self)  # resets hidden states for all LIF neurons in net
        
        if self.encoder:
            self.encoder_population.reset_hidden()

        self.leaky1.reset_hidden()
        self.recurrent.reset_hidden()
        self.leaky2.reset_hidden()
        rspk, rmem = self.recurrent.init_rleaky()
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
        i = 0
        for slice in data_permuted:
            if self.encoder:
               
                x = self.encoder_connection(slice)
                x, _ = self.encoder_population(x)
                
                if self.layer_loss is not None:
                    encoder_acc.append(x.clone().cpu())


                x = self.linear1(x)
                x, _ = self.leaky1(x)
                
                if self.layer_loss is not None:
                    layer1_acc.append(x.clone().cpu())   

            else:
                x = self.linear1(slice)
                x, _ = self.leaky1(x)
                
                if self.layer_loss is not None:
                    layer1_acc.append(x.clone().cpu())
                

            x = self.linear2(x)
            x = self.dropout_rec(x)
            
            rspk, rmem = self.recurrent(x, rspk, rmem)
            
            if self.layer_loss is not None:
                layer2_acc.append(rspk.clone().cpu())

            x = self.linear3(rspk)
            x = self.dropout_out(x)
            
            x, _ = self.leaky2(x)
            
            spk_rec.append(x)

        batch_out = torch.stack(spk_rec)

        if self.layer_loss is not None:
            if self.encoder:
                net_loss = self.layer_loss([torch.stack(layer1_acc), torch.stack(layer2_acc), torch.stack(encoder_acc)])
                del encoder_acc
            else:
                net_loss = self.layer_loss([torch.stack(layer1_acc), torch.stack(layer2_acc)])

            del layer1_acc
            del layer2_acc

            return batch_out, net_loss
        else:
            return batch_out
    
    def debug_init(self):

        self.spk_monitor = probe.OutputMonitor(self, instance = (snn.Leaky, snn.RLeaky))
        self.mem_monitor = probe.AttributeMonitor('mem', False, self, instance = (snn.Leaky))
        self.spk_monitor.enable()
        self.mem_monitor.enable()

    def debug_pause(self):
        self.spk_monitor.disable()
        self.mem_monitor.disable() 
    
    def  debug_start(self):
        self.spk_monitor.enable()
        self.mem_monitor.enable()

    def clear_monitor(self):
        self.spk_monitor.clear_recorded_data()
        self.mem_monitor.clear_recorded_data()
    
    def get_monitor_results(self):
        return self.spk_monitor, self.mem_monitor
    


    @staticmethod
    def gen_gaussian_distribution( len, mean, std, max=1.0):

        #extend the binning in oder to create 
        # the full gaussian and then cut the final part that goes to zeros
        bin = round(len*1.40)
        mean = round(mean*1.40)
        offset = round((bin - len)/2)
        x = torch.linspace(0, bin, bin)
        y = torch.exp(-((x - mean) ** 2) / (2 * std ** 2))*max #multiply for the actual treshold max 
        yy = y[offset:-offset]
        yy[yy < 0.01] = 0.1
        return yy
    
        
    def save_to_npz(self, path):
        
        w_scale = self.linear1.quant_weight_scale().detach().cpu().numpy()
        w_zero_point = self.linear1.quant_weight_zero_point().detach().cpu().numpy()

        linear1 = self.linear1.weight.data.detach().cpu().numpy()
        linear1_quant = self.linear1.quant_weight().int().detach().cpu().numpy()
        leaky1_betas = self.leaky1.beta.data.detach().cpu().numpy()
        leaky1_vth = self.leaky1.threshold.data.detach().cpu().numpy()
        linear2 = self.linear2.weight.data.detach().cpu().numpy()
        linear2_quant = self.linear2.quant_weight().int().detach().cpu().numpy()

        recurrent_betas = self.recurrent.beta.data.detach().cpu().numpy()
        recurrent_vth = self.recurrent.threshold.data.detach().cpu().numpy()
        input_dense, input_dense_quant, activation_betas, activation_vth, output_dense, output_dense_quant = self.recurrent.recurrent.to_npz()

        linear3 = self.linear3.weight.data.detach().cpu().numpy()
        linear3_quant = self.linear3.quant_weight().int().detach().cpu().numpy()
        leaky2_betas = self.leaky2.beta.data.detach().cpu().numpy()
        leaky2_vth = self.leaky2.threshold.data.detach().cpu().numpy()

        if self.encoder:
            encoder_connection = self.encoder_connection.weight.data.detach().cpu().numpy()
            encoder_population_betas = self.encoder_population.beta.data.detach().cpu().numpy()
            encoder_population_vth = self.encoder_population.threshold.data.detach().cpu().numpy()
            
            # np.savez_compressed(path,w_scale=w_scale, w_zero_point=w_zero_point, encoder_connection=encoder_connection, encoder_population_betas=encoder_population_betas,
            #                     encoder_population_vth=encoder_population_vth, linear1=linear1, leaky1_betas=leaky1_betas,
            #                     leaky1_vth=leaky1_vth, linear2=linear2, recurrent_betas=recurrent_betas, recurrent_vth=recurrent_vth,
            #                     input_dense=input_dense, activation_betas=activation_betas,activation_vth=activation_vth, output_dense=output_dense,
            #                     linear3=linear3, leaky2_betas=leaky2_betas, leaky2_vth=leaky2_vth)
            np.savez_compressed(path, w_scale=w_scale, w_zero_point=w_zero_point, encoder_connection=encoder_connection, encoder_population_betas=encoder_population_betas,
                                encoder_population_vth=encoder_population_vth, linear1=linear1, linear1_quant=linear1_quant, leaky1_betas=leaky1_betas,
                                leaky1_vth=leaky1_vth, linear2=linear2, linear2_quant=linear2_quant, recurrent_betas=recurrent_betas, recurrent_vth=recurrent_vth,
                                input_dense=input_dense, input_dense_quant=input_dense_quant, activation_betas=activation_betas, activation_vth=activation_vth,output_dense=output_dense,
                                output_dense_quant=output_dense_quant, linear3=linear3, linear3_quant=linear3_quant, leaky2_betas=leaky2_betas, leaky2_vth=leaky2_vth)
        else:
            # np.savez_compressed(path, w_scale=w_scale, w_zero_point=w_zero_point, linear1=linear1, leaky1_betas=leaky1_betas,
            #                     leaky1_vth=leaky1_vth, linear2=linear2, recurrent_betas=recurrent_betas, recurrent_vth=recurrent_vth,
            #                     input_dense=input_dense, activation_betas=activation_betas, activation_vth=activation_vth,output_dense=output_dense,
            #                     linear3=linear3, leaky2_betas=leaky2_betas, leaky2_vth=leaky2_vth)
            
            np.savez_compressed(path, w_scale=w_scale, w_zero_point=w_zero_point, linear1=linear1, linear1_quant=linear1_quant, leaky1_betas=leaky1_betas,
                                leaky1_vth=leaky1_vth, linear2=linear2, linear2_quant=linear2_quant, recurrent_betas=recurrent_betas, recurrent_vth=recurrent_vth,
                                input_dense=input_dense, input_dense_quant=input_dense_quant, activation_betas=activation_betas, activation_vth=activation_vth,output_dense=output_dense,
                                output_dense_quant=output_dense_quant, linear3=linear3, linear3_quant=linear3_quant, leaky2_betas=leaky2_betas, leaky2_vth=leaky2_vth)

    def from_npz(self, path):
       
        data = np.load(path,allow_pickle=True)
        if self.encoder:
            self.encoder_connection.weight.data = torch.tensor(data['encoder_connection'])
            self.encoder_population.beta.data = torch.tensor(data['encoder_population_betas' ])
            self.encoder_population.threshold.data = torch.tensor(data['encoder_population_vth'])
    
        self.linear1.weight.data = torch.tensor(data['linear1'])
        self.leaky1.beta.data = torch.tensor(data['leaky1_betas'])
        self.leaky1.threshold.data = torch.tensor(data['leaky1_vth'])

        self.linear2.weight.data = torch.tensor(data['linear2'])

        self.recurrent.beta.data = torch.tensor(data['recurrent_betas'])
        self.recurrent.threshold.data = torch.tensor(data['recurrent_vth'])

        self.recurrent.recurrent.from_npz(data['input_dense'], data['activation_betas'],data['activation_vth'], data['output_dense'])

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
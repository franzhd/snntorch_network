import torch
import torch.nn as nn
import snntorch as snn
import brevitas.nn as qnn 
from snntorch import utils
from snntorch import surrogate
import torch.nn.functional as F
from snntorch import functional as SF
from snntorch.functional import quant
from RecurrentAHPC_debug import QuantRecurrentAhpc
import numpy as np
import pdb 

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
                vth_in, vth_recurrent, vth_out,
                beta_in, beta_recurrent, beta_back, beta_out,
                encoder_dim=None, vth_enc_value=1.0, vth_std=1000,beta_std=1000,
                drop_recurrent=0.0, drop_back=0.0, drop_out=0.0,
                state_quant=False,
                time_dim=1,
                num_bits=8):
        
        super(QuantAhpcNetwork, self).__init__()


        self.time_dim = time_dim
        
        if state_quant:
            self.quant = False
            # self.quant = quant.state_quant(num_bits=17, uniform=True, threshold=threshold)
        else:
            self.quant = False

        if encoder_dim is not None:
            self.encoder = True
            vth_e = self.gen_gaussian_distribution(encoder_dim, encoder_dim/2, vth_std, vth_enc_value)
            beta_e = self.gen_gaussian_distribution(encoder_dim, encoder_dim/2, beta_std)
            
            self.encoder_connection = qnn.QuantLinear(num_inputs, encoder_dim, bias=False,
                                            weight_bit_width=num_bits)

           
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
                                           weight_bit_width=num_bits)

           

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

        self.recurrent = QuantRecurrentAhpc(beta_recurrent, beta_back, spike_grad=grad, linear_features = num_hidden_2,
                                            init_hidden=True, reset_delay=True, learn_beta=True,
                                            learn_threshold=True, learn_recurrent=True, 
                                            threshold=vth_recurrent,
                                            shared_weight_quant=self.linear1.weight_quant,state_quant=self.quant, dropout=drop_back)
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
        utils.reset(self)  # resets hidden states for all LIF neurons in net
        dims = list(range(data.dim()))  # Creates a list of dimensions
        dims.pop(self.time_dim)                     # Remove the selected dimension
        dims.insert(0, self.time_dim)               # Insert the selected dimension at the front
        data_permuted = data.permute(dims)  # Permute the tensor
        for slice in data_permuted:
            if self.encoder:
                x = self.encoder_connection(slice)
                x, _ = self.encoder_population(x)
                x = self.linear1(x)
                x, _ = self.leaky1(x)
            else:
                x = self.linear1(slice)
                x, _ = self.leaky1(x)

            x = self.linear2(x)
            x = self.dropout_rec(x)
            spk1 = self.recurrent(x)
            x = self.linear3(spk1)
            x = self.dropout_out(x)
            x, _ = self.leaky2(x)
            spk_rec.append(x)

        batch_out = torch.stack(spk_rec)

        return batch_out
    
    @staticmethod
    def gen_gaussian_distribution( len, mean, std, max=1.0):

        #extend the binning in oder to create 
        # the full gaussian and then cut the final part that goes to zeros
        bin = round(len*1.40)
        mean = round(mean*1.40)
        offset = round((bin - len)/2)
        x = torch.linspace(0, bin, bin)
        y = torch.exp(-((x - mean) ** 2) / (2 * std ** 2))*max #multiply for the actual treshold max 0.5 is placeolder
        yy = y[offset:-offset]
        yy[yy < 0.01] = 0.1
        return yy
    

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
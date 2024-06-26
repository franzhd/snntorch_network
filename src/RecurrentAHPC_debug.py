import snntorch as snn
import torch.nn as nn
import brevitas.nn as qnn
from snntorch.functional import quant
from brevitas.quant import Int8WeightPerTensorFixedPoint
import torch 

class QuantRecurrentAhpc(snn.RLeaky):

    def __init__(
        self,
        beta,
        back_beta,
        shared_weight_quant = Int8WeightPerTensorFixedPoint,
        linear_features=None,
        kernel_size=None,
        threshold=1.0,
        back_vth= 10000.0,
        back_grad=None,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        learn_recurrent=True, 
        reset_mechanism="zero",
        state_quant=False,
        output=False,
        reset_delay=False,
        alpha = None,
        dropout=0.0,
        
    ):
        
        super(QuantRecurrentAhpc, self).__init__(
            beta=beta,
            all_to_all=True,
            linear_features=linear_features,
            kernel_size=kernel_size,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            init_hidden=init_hidden,
            inhibition=inhibition,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            learn_recurrent=learn_recurrent,
            reset_mechanism=reset_mechanism,
            state_quant=state_quant,
            output=output,
            reset_delay=reset_delay,
        )

        assert linear_features is not None, "Linear features must not be None"

        self.back_grad = back_grad if back_grad is not None else spike_grad

        if self.state_quant is not False:
            self.back_quant = False
            #self.back_quant = quant.state_quant(num_bits=17, uniform=True, threshold=back_vth)
        else:
            self.back_quant = False

        self.alpha = alpha
        self.back_beta = back_beta
        self.back_vth = back_vth
        self.dropout = dropout
        self.shared_weight_quant = shared_weight_quant
        self.overwrite_self_recurrent()
        self.recurrent.init_ahpc()  


        
    def overwrite_self_recurrent(self):
        self.recurrent = QuantAhpcBlock(self.back_beta,
                                        self.back_vth, 
                                        self.back_grad, 
                                        self.linear_features, 
                                        shared_weight_quant=self.shared_weight_quant,
                                        state_quant=self.back_quant,
                                        dropout=self.dropout)
                                        
    
    def _base_state_function_hidden(self, input_):
        base_fn = (
            self.beta.clamp(0, 1) * self.mem
            + input_
            - self.recurrent(self.spk)
        )
        return base_fn
    
    def reset_hidden(self):
        super().reset_hidden()
        self.recurrent.reset_hidden()



class QuantAhpcBlock(nn.Module):
    def __init__(self, beta, vth, grad, features, shared_weight_quant, num_bits=8, state_quant=False, dropout=0.0):

        super(QuantAhpcBlock, self).__init__()
        
        self.input_dense = qnn.QuantLinear(features, features, bias=False,
                                           weight_quant= shared_weight_quant,
                                           weight_bit_width=num_bits)
        
        self.dropout = nn.Dropout(dropout)  
        
        self.activation = snn.Leaky(beta=beta, 
                                    spike_grad=grad,
                                    threshold= vth, 
                                    learn_threshold=True, 
                                    learn_beta=True, 
                                    reset_mechanism='zero',
                                    reset_delay=False,
                                    state_quant=state_quant)
         
        self.output_dense = qnn.QuantLinear(features, features, bias=False,
                                      weight_quant= shared_weight_quant,
                                      weight_bit_width=num_bits)
    def reset_hidden(self):
        self.activation.reset_hidden()    
    
    def init_ahpc(self): 
            self.activation.init_leaky()


    def forward(self, x):
        # Add your forward pass implementation here
        x = self.input_dense(x)
        x = self.dropout(x)
        _, mem_out = self.activation(x)
        x = self.output_dense(mem_out)
        return x
    
    def to_npz(self):
        input_dense = self.input_dense.weight.detach().cpu().numpy()
        activation_beta = self.activation.beta.data.detach().cpu().numpy()
        output_dense = self.output_dense.weight.detach().cpu().numpy()
        return input_dense, activation_beta, output_dense
    
    def from_npz(self, input_dense, activation_beta, output_dense):
        
        self.input_dense.weight.data = torch.tensor(input_dense)
        self.activation.beta.data = torch.tensor(activation_beta)   
        self.output_dense.weight.data = torch.tensor(output_dense)


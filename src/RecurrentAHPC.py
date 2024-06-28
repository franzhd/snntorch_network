import snntorch as snn
import torch.nn as nn
import brevitas.nn as qnn
import snntorch.functional.quant as quant
from brevitas.quant import Int8WeightPerTensorFixedPoint
import torch 


class RecurrentAhpc(snn.RLeaky):

    def __init__(
        self,
        beta,
        back_beta,
        linear_features=None,
        kernel_size=None,
        threshold=1.0,
        back_threshold= 10000.0,
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
    ):
        assert linear_features is not None, "Linear features must not be None"

        self.recurrent_type = 'leaky' if alpha is None else 'synaptic'
        self.back_grad = back_grad if back_grad is not None else spike_grad
        
        self.alpha = alpha
        self.back_beta = back_beta
        self.back_grad = back_grad
        self.back_threshold = back_threshold

        super(RecurrentAhpc, self).__init__(
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

        
    def _init_recurrent_linear(self):
        self.recurrent = AhpcBlock(self.back_beta,
                                   self.back_threshold, 
                                   self.back_grad, 
                                   self.linear_features, 
                                   recurrent_type=self.recurrent_type,
                                   alpha=self.alpha)
    
    def _base_state_function_hidden(self, input_):
        base_fn = (
            self.beta.clamp(0, 1) * self.mem
            + input_
            - self.recurrent(self.spk)
        )
        return base_fn

class AhpcBlock(nn.Module):
    def __init__(self, beta, threshold, grad, features, recurrent_type='leaky', alpha=None):

        super(AhpcBlock, self).__init__()
        self.first_dense = nn.Linear(features, features)
        
        if recurrent_type == 'leaky':
            
            self.activation = snn.Leaky(beta=beta, 
                                        spike_grad=grad,
                                        threshold= threshold, 
                                        learn_threshold=True, 
                                        learn_beta=True, 
                                        reset_mechanism='zero',
                                        reset_delay=False)
            
        if recurrent_type =='synaptic' and alpha is not None:

            self.activation = snn.Synaptic(alpha=alpha,
                                           beta=beta, 
                                           spike_grad=grad,
                                           threshold= threshold, 
                                           learn_beta=True,
                                           learn_threshold=True, 
                                           reset_mechanism='zero',
                                           reset_delay=False)
            
        self.output = nn.Linear(features, features)

                                        
    def forward(self, x):
        # Add your forward pass implementation here
        x = self.first_dense(x)
        _, mem_out = self.activation(x)
        x = self.output(mem_out)
        return x

class QuantRecurrentAhpc(snn.RLeaky):

    def __init__(
        self,
        beta,
        back_beta,
        shared_weight_quant = Int8WeightPerTensorFixedPoint,
        linear_features=None,
        kernel_size=None,
        threshold=1.0,
        back_threshold= 10000.0,
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

        self.recurrent_type = 'leaky' if alpha is None else 'synaptic'
        self.back_grad = back_grad if back_grad is not None else spike_grad


        self.alpha = alpha
        self.back_beta = back_beta
        self.back_threshold = back_threshold

        self.shared_weight_quant = shared_weight_quant
        self.overwrite_self_recurrent()


        
    def overwrite_self_recurrent(self):
        self.recurrent = QuantAhpcBlock(self.back_beta,
                                        self.back_threshold, 
                                        self.back_grad, 
                                        self.linear_features, 
                                        recurrent_type=self.recurrent_type,
                                        alpha=self.alpha,
                                        shared_weight_quant=self.shared_weight_quant)
    
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
    def __init__(self, beta, threshold, grad, features, shared_weight_quant, num_bits=8, recurrent_type='leaky', alpha=None, state_quant=False):

        super(QuantAhpcBlock, self).__init__()
        
        self.recurrent_type = recurrent_type

        self.input_dense = qnn.QuantLinear(features, features, bias=False,
                                           weight_quant= shared_weight_quant,
                                           weight_bit_width=num_bits)
        
        # if recurrent_type == 'leaky':
            
        self.activation = snn.Leaky(beta=beta, 
                                    spike_grad=grad,
                                    threshold= threshold, 
                                    learn_threshold=True, 
                                    learn_beta=True, 
                                    reset_mechanism='none',
                                    reset_delay=False,
                                    state_quant=state_quant)
    
        # if recurrent_type =='synaptic' and alpha is not None:

        #     self.activation = snn.Synaptic(alpha=alpha,
        #                                    beta=beta, 
        #                                    spike_grad=grad,
        #                                    threshold= threshold, 
        #                                    learn_beta=True,
        #                                    learn_alpha=True,
        #                                    learn_threshold=True, 
        #                                    reset_mechanism='zero',
        #                                    reset_delay=False)
            
        self.output_dense = qnn.QuantLinear(features, features, bias=False,
                                      weight_quant= shared_weight_quant,
                                      weight_bit_width=num_bits)
    def reset_hidden(self):
        self.activation.reset_hidden()    
    
    def forward(self, x):
        # Add your forward pass implementation here
        x = self.input_dense(x)
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


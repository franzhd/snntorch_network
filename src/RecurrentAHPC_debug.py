import snntorch as snn
from snntorch._neurons.neurons import _SpikeTensor, _SpikeTorchConv
import torch.nn as nn
import brevitas.nn as qnn
from snntorch.functional import quant
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint
import torch 
import numpy as np

class QuantRecurrentAhpc(snn.RLeaky):

    def __init__(
        self,
        beta,
        back_beta,
        back_vth,
        shared_weight_quant = Int8WeightPerTensorFixedPoint,
        linear_features=None,
        kernel_size=None,
        threshold=1.0,
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
        dropout=0.0
        
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
            reset_delay=reset_delay
        )

        assert linear_features is not None, "Linear features must not be None"

        self.back_grad = back_grad if back_grad is not None else spike_grad

        if not state_quant:
            self.back_quant = False
        else:
            self.back_quant = True

        self.alpha = alpha
        self.back_beta = back_beta
        self.back_vth = back_vth
        self.dropout = dropout
        self.shared_weight_quant = shared_weight_quant
        self.overwrite_self_recurrent()
    


        
    def overwrite_self_recurrent(self):
        self.recurrent = QuantAhpcBlock(self.back_beta,
                                        self.back_vth, 
                                        self.back_grad, 
                                        self.linear_features, 
                                        shared_weight_quant=self.shared_weight_quant,
                                        state_quant=self.back_quant,
                                        dropout=self.dropout)
                                        
    def _build_state_function_hidden(self, input_):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = (
                self._base_state_function_hidden(input_)
                - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            # print("reset to zero step")
            # state_fn = self._base_state_function_hidden(
            #     input_
            # ) - self.reset * self._base_state_function_hidden(input_)
            # print("reset to zero step")
            state_fn = (1.0-self.reset) * self._base_state_function_hidden(input_)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function_hidden(input_)
        return state_fn
    
    def _build_state_function(self, input_, spk, mem):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = self._base_state_function(
                input_, spk, mem - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            # state_fn = self._base_state_function(
            #     input_, spk, mem
            # ) - self.reset * self._base_state_function(input_, spk, mem)
            state_fn = (1.0-self.reset) * self._base_state_function(input_, spk, mem)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, spk, mem)

        return state_fn
    def _base_state_function_hidden(self, input_):
        base_fn = (
            self.beta.clamp(0, 1) * self.mem
            + input_
            - self.recurrent(self.spk)
        )
        return base_fn
    
    def _base_state_function(self, input_, spk, mem):
        base_fn = self.beta.clamp(0, 1) * mem + input_ - self.recurrent(spk)
        return base_fn
    
    def reset_hidden(self):
        super().reset_hidden()
        self.recurrent.reset_hidden()

    def forward(self, input_, spk=False, mem=False):
        if hasattr(spk, "init_flag") or hasattr(
            mem, "init_flag"
        ):  # only triggered on first-pass
            spk, mem = _SpikeTorchConv(spk, mem, input_=input_)
        # init_hidden case
        elif mem is False and hasattr(self.mem, "init_flag"):
            self.spk, self.mem = _SpikeTorchConv(
                self.spk, self.mem, input_=input_
            )
        # TO-DO: alternatively, we could do torch.exp(-1 /
        # self.beta.clamp_min(0)), giving actual time constants instead of
        # values in [0, 1] as initial beta beta = self.beta.clamp(0, 1)

        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            mem = self._build_state_function(input_, spk, mem)

            if self.state_quant:
                mem = self.state_quant(mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)  # batch_size
            else:
                spk = self.fire(mem)

            if not self.reset_delay:
                do_reset = spk / self.graded_spikes_factor - self.reset  # avoid double reset
                if self.reset_mechanism_val == 0:  # reset by subtraction
                    mem = mem - do_reset * self.threshold
                elif self.reset_mechanism_val == 1:  # reset to zero
                    mem = mem - do_reset * mem

            return spk, mem

        # intended for truncated-BPTT where instance variables are hidden
        # states
        if self.init_hidden:
            self._rleaky_forward_cases(spk, mem)
            self.reset = self.mem_reset(self.mem)
            self.mem = self._build_state_function_hidden(input_)

            if self.state_quant:
                self.mem = self.state_quant(self.mem)
            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)


            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk




class QuantAhpcBlock(nn.Module):
    def __init__(self, beta, vth, grad, features, shared_weight_quant, num_bits=8, state_quant=False, dropout=0.0, delay=True):

        super(QuantAhpcBlock, self).__init__()
        
        self.input_dense = qnn.QuantLinear(features, features, bias=False,
                                           weight_quant= shared_weight_quant,
                                           weight_bit_width=num_bits)
        
        self.dropout = nn.Dropout(dropout)  
        self.activation_quant = qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, act_bit_width=24, return_quant_tensor=True)
        if not state_quant:

            self.activation = snn.Leaky(beta=beta, 
                                        spike_grad=grad,
                                        threshold= vth, 
                                        learn_threshold=True, 
                                        learn_beta=True, 
                                        reset_mechanism='zero',
                                        reset_delay=False)
        else:
            activativation_quant = quant.state_quant(num_bits=16, threshold=vth)
            self.activation = snn.Leaky(beta=beta, 
                                        spike_grad=grad,
                                        threshold= vth, 
                                        learn_threshold=True, 
                                        learn_beta=True, 
                                        reset_mechanism='zero',
                                        reset_delay=False,
                                        state_quant=activativation_quant)
         
        self.output_dense = qnn.QuantLinear(features, features, bias=False,
                                      weight_quant= shared_weight_quant,
                                      weight_bit_width=num_bits)
        
        self.out_quant = qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, act_bit_width=24, return_quant_tensor=True)

        if delay:
            self.features = features
            self.accumulator = torch.zeros((1,features))
        else:
            self.features = None

    def reset_hidden(self):
        if self.features is not None:
            self.accumulator = torch.zeros((1,self.features))
        self.activation.reset_hidden()    
    

    def forward(self, input):
        # Add your forward pass implementation here
        if self.features is not None:
            x = self.input_dense(self.accumulator.to(input.device))
            self.accumulator = input
        else:
            x = self.input_dense(input)
        x = self.dropout(x)
        x = self.activation_quant(x)
        spk , _ = self.activation(x)
        # if self.features is not None:
        #     out = self.accumulator.to(x.device)
        #     self.accumulator = self.output_dense(spk) 
        # else:
        out = self.output_dense(spk)
        out = self.out_quant(out)

        return out
    
    def to_npz(self):
        input_dense = self.input_dense.weight.detach().cpu().numpy()
        input_dense_quant = self.input_dense.quant_weight().value.detach().cpu().numpy()

        activation_beta = self.activation.beta.data.detach().cpu().numpy()
        activation_vth = self.activation.threshold.data.detach().cpu().numpy()

        output_dense = self.output_dense.weight.detach().cpu().numpy()
        output_dense_quant = self.output_dense.quant_weight().value.detach().cpu().numpy()

        return input_dense,input_dense_quant, \
                activation_beta, activation_vth, \
                output_dense, output_dense_quant
    
    def from_npz(self, input_dense, activation_beta, activation_vth, output_dense):
        
        self.input_dense.weight.data = torch.tensor(input_dense)
        self.activation.beta.data = torch.tensor(activation_beta)   
        self.activation.threshold.data = torch.tensor(activation_vth)
        self.output_dense.weight.data = torch.tensor(output_dense)


from easydict import EasyDict as edict
import torch.nn as nn
from ..base_config import BaseConfig, SequentialModelConfig


class Linear_Config(SequentialModelConfig):
    """
    Configuration for 
    - skill prior module (state : vector)
    - Low-level skill decoder module (closed loop)  XXXX 
    """
    def __init__(self, config):

        self.set_attrs(config)
        self.name = "MLP Configuration for skill encoder & skill decoder"
        self.get_build()
        
    def get_build(self):
        build = self.layerbuild(["block_cls", "in_feature", "hidden_dim", None, "act_cls", "bias", "dropout"])
        build += self.layerbuild(["block_cls", "hidden_dim", "hidden_dim", "norm_cls", "act_cls"], self.get("n_blocks"))
        build += self.layerbuild(["block_cls", "hidden_dim", "out_dim", None, None,  "bias", "dropout"])
        self.build = build


# skill encoder configuration 
class RNN_Config(SequentialModelConfig):
    """
    Configuration for 
    - skill encoder (posterior)
    """
    def __init__(self, config):
        super().__init__()
        self.set_attrs(config)
        self.name = "RNN Encoder with projection for skill encoder"
        self.get_build()

    def get_build(self):
        # build = self.layerbuild(["linear_cls", "in_feature", "hidden_dim", None, "act_cls", "bias"])
        # build += self.layerbuild(["rnn_cls", "hidden_dim", "hidden_dim", "n_blocks", "bias", "batch_first", "dropout"])
        # build += self.layerbuild(["linear_cls", "hidden_dim", "out_dim", None, None, "bias"])
        
        build = self.layerbuild(["linear_cls", "in_feature", "hidden_dim", None, "act_cls", "bias"])
        build += self.layerbuild(["rnn_cls", "hidden_dim", "hidden_dim", "n_blocks", "bias", "batch_first", "dropout"])
        build += self.layerbuild(["linear_cls", "hidden_dim", "out_dim", None, None, "false"])
        self.build = build



class TransformerModelConfig(BaseConfig):
    def __init__(self, config):
        super().__init__()
        self.set_attrs(config)
        self.name = "Transformer Configuration for skill prior & skill encoder"
        self.get_build()

    def layerbuild(self, attr_list, repeat = None):
        build =  [[ self.get(attr_nm) for attr_nm in attr_list  ]]
        if repeat is not None:
            build = build * repeat
        return build 

    def get_build(self):
        build = self.layerbuild(["block_cls", "layer_cls", "hidden_dim", "nhead", "ff_dim", "true"])[0]
        self.build = build

# class Conv_Config(BaseModelConfig):
#     """
#     Configuration for 
#     - skill prior module (state : image)    
#     - Low-level skill decoder module (closed loop)
#     """
#     def __init__(self):
#         self.set_attr(
#             edict(
#                 n_blocks = 6,
#                 in_feature = 3,
#                 hidden_dims = [8,16,32],
#                 kernel_size = 3,
#                 out_dim = 32,
#                 norm_cls = nn.BatchNorm1d,
#                 act_cls = nn.LeakyReLU
#             )
#         )
#         self.get_build()
        
#     def get_build(self):
#         channels = [self.in_feature] + self.hidden_dims

#         build = []
#         for i in range( len(channels) - 1 ):
#             build += [[channels[i], channels[i+1], self.kernel, self.norm_cls, self.act_cls]]
#         self.build = build




import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
	
    def forward(self, input, shape):
        output = input.view(*shape)		
        return output


import torch

STATE_MT = "mt"
STATE_VT = "vt" 
STATE_EF = 'error_feedback'
STATE_BUFFER_INDICES = 'I'
STATE_BUFFER_VALUES = 'V'
STATE_GRADIENT_DENSITY = 'gradient_density'
STATE_BUFFER_INDEX = 'buffer_index'
STATE_GRAD_BUFFER = 'gradient_buffer'

def compute_standard_topk(x, k):
    indices = torch.topk(
        input=x.abs().view(-1),
        k=k,
        sorted=False
    ).indices
    values = x.view(-1)[indices]
    return indices, values

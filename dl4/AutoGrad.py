import torch
from torch.autograd import Function

class ExpAddCos(Function):
    @staticmethod
    def forward(ctx, x, y):
        exp_x = torch.exp(x)
        cos_y = torch.cos(y)
        ctx.save_for_backward(x, y, exp_x)

        return exp_x + cos_y

    @staticmethod
    def backward(ctx, grad_output):
        x, y, exp_x = ctx.saved_tensors
        grad_x = grad_output * exp_x
        grad_y = grad_output * (-torch.sin(y))
        
        return grad_x, grad_y


def test_exp_add_cos():
    device = torch.device('cpu')
    
    x = torch.randn(3, requires_grad=True, device=device)
    y = torch.randn(3, requires_grad=True, device=device)
    
    result_custom = ExpAddCos.apply(x, y)
    
    result_torch = torch.exp(x) + torch.cos(y)
    
    print("fprop is the same: ", torch.allclose(result_custom, result_torch, atol=1e-6))
    
    result_custom.sum().backward()
    grad_x_custom = x.grad.clone()
    grad_y_custom = y.grad.clone()
    
    x.grad = None
    y.grad = None
    
    result_torch.sum().backward()
    grad_x_ref = x.grad.clone()
    grad_y_ref = y.grad.clone()
    
    print("X grad is the same: ", torch.allclose(grad_x_custom, grad_x_ref, atol=1e-6))
    print("Y grad is the same: ", torch.allclose(grad_y_custom, grad_y_ref, atol=1e-6))

test_exp_add_cos()
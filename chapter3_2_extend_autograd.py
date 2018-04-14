from torch.autograd import Function
import torch

class MultiplyAdd(Function):

    @staticmethod
    def forward(ctx, w, x, b):
        print('type in forward', type(x))
        ctx.save_for_backward(w, x)
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_variables
        print('type in backward', type(x))
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        return grad_w, grad_x, grad_b



class Sigmoid(Function):

    @staticmethod
    def forward(ctx, x):
        output = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, _ = ctx.saved_variables
        grad_x = output + (1 - output) * grad_output
        return grad_x
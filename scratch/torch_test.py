from scratch.utils import *


def FT(shape):
    return torch.randn(shape)


def flatten_tensors(tensors):
    """
    Flatten tensors into a single contiguous 1D buffer
    https://github.com/pytorch/pytorch/blob/master/torch/_utils.py
    """
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    numels = [tensor.numel() for tensor in tensors]
    size = sum(numels)
    offset = 0
    flat = tensors[0].new(size)
    for tensor, numel in zip(tensors, numels):
        flat.narrow(0, offset, numel).copy_(tensor, broadcast=False)
        offset += numel
    return flat


def unflatten_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors"""
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def test_flatten_unflatten():
    a, b, c = FT([2,3]), FT([3,2]), FT([4,4])
    print(a)
    print(b)
    print(c)
    T = flatten_tensors([a,b,c])
    print(T)
    print(unflatten_tensors(T, [a,b,c]))



def get_qnet():
    return FFQfunc(input_shape=[2,3],
               action_dim=4,
               convs=[],
               fc_hidden_sizes=[8],
               dueling=False)

x = Variable(FT([7,2,3]))
q1 = get_qnet()
q2 = get_qnet()
print(q1(x))

print(len(q1.parameters_to_binary()))
q2.parameters_from_binary(q1.parameters_to_binary())
print(q2(x))


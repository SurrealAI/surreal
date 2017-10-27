from .notifier import PSNotifier


class TorchPSNotifier(PSNotifier):
    def flatten_tensors(self, tensors):
        """Flatten tensors into a single contiguous 1D buffer"""
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

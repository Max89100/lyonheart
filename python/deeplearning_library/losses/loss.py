from .._deeplearning_core import CoreTensor

class Loss:
    def __call__(self, y_pred, y_true):
        raise NotImplementedError

class MSELoss(Loss):
    def __call__(self, y_pred: CoreTensor, y_target: CoreTensor) -> CoreTensor:
        diff = (y_pred - y_target).pow_scalar(2.0)
        return diff.mean()

class CrossEntropyLoss(Loss):
    def __call__(self, y_pred: CoreTensor, y_target: CoreTensor):
        eps  = 1e-10
        return (y_pred.add_scalar(eps).log() * y_target).sum_dim(1).neg().mean()

class LogSoftmax(Loss):
    '''Applies Softmax and CrossEntropy on logits. Remove Softmax layer from your model if you're using this loss function.'''
    def __call__(self, logits: CoreTensor, y_target: CoreTensor):
        m = logits.max_dim(1)
        shifted_logits = logits - m
        log_sum_exp = shifted_logits.exp().sum_dim(1).log()
        log_probs = shifted_logits - log_sum_exp
        return (log_probs * y_target).sum_dim(1).neg().mean()
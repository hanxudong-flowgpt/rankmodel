from baseconfig import *
from seqconfig import *


class MultimodalConfig(SequenceModelConfig):
  """Wrapper class for model hyperparameters."""

  def __init__(self, **kwargs):
    super(MultimodalConfig, self).__init__(**kwargs)

    self.atten_gate = "softmax"  # sigmoid
    self.atten_pooling = "concat"  # sum

    self.num_heads = 32

    # Update other configuration parameters.
    self.__dict__.update(kwargs)
    self.cvt_func_map = dict([(k, str2typefunc(v)) for k, v in self.__dict__.items() if k != ''])

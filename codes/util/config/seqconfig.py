from baseconfig import *


class SequenceModelConfig(ModelConfig):
  """Wrapper class for model hyperparameters."""

  def __init__(self, **kwargs):
    super(SequenceModelConfig, self).__init__(**kwargs)

    # Sequence combiner. 'mean' 'sum' 'concat'
    self.sequence_item_combiner = "transformer"
    self.use_self_attention = True
    self.use_context_attention = True
    self.use_query_attention = True
    self.use_target_attention = False
    # if use_self_attention is true && use_context|query|target_attention are both false, sequence_atten_combiner must be set
    self.sequence_atten_combiner = "mean"
    # queryseq
    self.queryseq_combiner = "concat"

    # layers should be shared
    self.share_emdedding = True
    self.share_user_layer = True

    # only for recall
    self.share_user_vec = False
    self.share_item_vec = True

    # Hidden Units after concat item sequence
    self.sequence_item_hidden_layer = [512, 256]
    self.seqs_self_atten_hidden_units = [340, 340]

    self.seqs_context_atten_hidden_units = [32, 1]
    self.seqs_query_atten_hidden_units = [64, 1]

    self.i2i_num_units = 24

    self.attentions = ["usertag", "queryseg_norm"]
    self.sa_num_units = 128
    self.ua_num_units = 128

    self.sa_num_output_units = 128
    self.ua_num_output_units = 128

    self.num_heads = 8

    self.ip_use_constant = True
    self.ip_constant = 6.5

    # Update other configuration parameters.
    self.__dict__.update(kwargs)
    #self.cvt_func_map = dict([(k, str2typefunc(v)) for k, v in self.__dict__.items() if k != ''])

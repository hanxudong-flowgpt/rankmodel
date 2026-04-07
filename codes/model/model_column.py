import json
from collections import OrderedDict

USER_COLUMNS = "user_columns"
QUERY_COLUMNS = "query_columns"
ITEM_COLUMNS = "item_columns"
UI_COLUMNS = "ui_columns"
WIDE_COLUMNS = "wide_columns"

I2I_SEQUENCE_COLUMNS = "i2i_sequence_columns"
CROSS_COLUMNS = "cross_columns"

QUERYSEQ_COLUMNS = "queryseq_columns"

TITLE_COLUMNS = "title_columns"

SEQUENCE_COLUMNS = "sequence_columns"
SEQ_LENGTH_COLUMNS = "seq_length_columns"
SEQ_CONTEXT_COLUMNS = "seq_context_columns"

SEQ_CONTEXT_ATTEN_COLUMNS = "seq_context_atten_columns"
QUERY_ATTEN_COLUMNS = "query_atten_columns"
USER_ATTEN_COLUMNS = "user_atten_columns"

SEQUENCE_MULTIMODAL = "sequence_multimodal"
PERSONAL_COLUMNS = "personal_columns"

SEQUENCE_CONTEXT_COLUMNS = "sequence_context_columns"

COLUMN_NAME = "column_name"


def merge_model_column(a, b, is_local=False):
    dict_a = a.config_dict.copy()
    dict_b = b.config_dict.copy()
    for k, v in dict_b.items():
        exists = []
        for x in dict_a[k]:
            exists.append(x["column_name"])
        for x in v:
            if x["column_name"] not in exists:
                dict_a[k].append({"column_name": x["column_name"]})
    return ModelColumns(dict_a, is_local)


class ModelColumns:
    def __init__(self, conf_file_path, is_local=False):
        self.is_local = is_local
        if type(conf_file_path) is not dict:
            self.conf_file_path = conf_file_path
            with open(self.conf_file_path) as f:
                self._config = json.load(f)
        else:
            self._config = conf_file_path
        # print(conf_file_path, self._config)
        self.fav_seq_column_dict = self.to_columns_dict("fav_seq_columns")
        self.buy_seq_column_dict = self.to_columns_dict("buy_seq_columns")
        self.clk_seq_column_dict = self.to_columns_dict("clk_seq_columns")
        self.user_column_dict = self.to_columns_dict(USER_COLUMNS)
        self.query_column_dict = self.to_columns_dict(QUERY_COLUMNS)
        self.item_column_dict = self.to_columns_dict(ITEM_COLUMNS)
        self.ui_column_dict = self.to_columns_dict(UI_COLUMNS)
        self.wide_column_dict = self.to_columns_dict(WIDE_COLUMNS)

        self.queryseq_column_dict = self.to_columns_dict(QUERYSEQ_COLUMNS)
        self.title_column_dict = self.to_columns_dict(TITLE_COLUMNS)

        self.sequence_column_dict = self.to_columns_dict(SEQUENCE_COLUMNS)
        self.seq_length_column_dict = self.to_columns_dict(SEQ_LENGTH_COLUMNS)
        self.seq_context_column_dict = self.to_columns_dict(SEQ_CONTEXT_COLUMNS)

        self.query_atten_column_dict = self.to_columns_dict(QUERY_ATTEN_COLUMNS)
        self.user_atten_column_dict = self.to_columns_dict(USER_ATTEN_COLUMNS)
        self.seq_context_atten_column_dict = self.to_columns_dict(SEQ_CONTEXT_ATTEN_COLUMNS)

        self.i2i_sequence_columns_dict = self.to_columns_dict(I2I_SEQUENCE_COLUMNS)
        self.cross_cloumns_dict = self.to_columns_dict(CROSS_COLUMNS)

        # self.multimodal = self._config[SEQUENCE_MULTIMODAL]
        # self.personal_columns_dict = self.to_columns_dict(PERSONAL_COLUMNS)

        self.column_groups = {}
        for k, v in self._config.items():
            if isinstance(v, list) and len(v) > 0:
                self.column_groups[k] = self.to_columns_dict(k).keys()

    def to_columns_dict(self, column_block_name):
        column_block_dict = OrderedDict()
        if column_block_name not in self._config:
            return column_block_dict
        for column in self._config[column_block_name]:
            if COLUMN_NAME in column:
                column_block_dict[column[COLUMN_NAME]] = column
            if self.is_local:
                break
        return column_block_dict

    @property
    def all_columns_in_need(self):
        x = []
        for xx in self.column_groups.values():
            print(xx)
            x.extend(xx)
            print("after extend:", x)
        return x

    @property
    def config_dict(self):
        return self._config

    @property
    def simple_dict(self):
        conf = {}
        for k, v in self._config.items():
            if len(v) == 0:
                conf[k] = []
            else:
                if self.is_local:
                    conf[k] = [v[-1][COLUMN_NAME]]
                else:
                    conf[k] = []
                    for x in v:
                        if COLUMN_NAME in x:
                            conf[k].append(x[COLUMN_NAME])
        return conf

    @property
    def user_columns(self):
        return self.user_column_dict.keys()

    @property
    def query_columns(self):
        return self.query_column_dict.keys()

    @property
    def item_columns(self):
        return self.item_column_dict.keys()

    @property
    def ui_columns(self):
        return self.ui_column_dict.keys()

    @property
    def wide_columns(self):
        return self.wide_column_dict.keys()

    @property
    def title_columns(self):
        return self.title_column_dict.keys()

    @property
    def queryseq_columns(self):
        return self.queryseq_column_dict.keys()

    @property
    def sequence_columns(self):
        return self.sequence_column_dict.keys()

    @property
    def multimodal_columns(self):
        return self.multimodal

    @property
    def seq_context_columns(self):
        return self.seq_context_column_dict.keys()

    @property
    def seq_context_atten_columns(self):
        return self.seq_context_atten_column_dict.keys()

    @property
    def query_atten_columns(self):
        return self.query_atten_column_dict.keys()

    @property
    def user_atten_columns(self):
        return self.user_atten_column_dict.keys()

    @property
    def seq_length_columns(self):
        return self.seq_length_column_dict.keys()

    @property
    def i2i_sequence_columns(self):
        return self.i2i_sequence_columns_dict.keys()

    @property
    def cross_columns(self):
        return self.cross_cloumns_dict.keys()

    @property
    def personal_columns(self):
        return self.personal_columns_dict.keys()

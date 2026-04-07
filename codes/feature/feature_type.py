# ****************************************************************#
# Author: qiaohan.qh@alibaba-inc.com
# Create Date: 2018-03-18 17:50
# Modify Author: qiaohan.qh@alibaba-inc.com
# Modify Date: 2018-03-18 17:50
# Function:
# ***************************************************************#
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class LongSequenceFeature(object):
    def __init__(self, seqname):
        self.id_names = []
        self.id_tensors = []
        # self.data_idx_ids = []

        self.pt_names = []
        self.pt_tensors = []
        self.mask_names = []
        self.mask_tensors = []

        self.group_vectors = {}
        self.group_names = []
        self.group_embedsize = []

        self.feaconf = {}
        self.seqname = seqname

    def get_fgconf_byname(self, name):
        return self.feaconf[name]

    def add_idfeature(self, tensor, groupname, fname, fconf):
        self.feaconf[fname] = fconf
        self.id_names.append(fname)
        self.id_tensors.append(tensor)

        if groupname not in self.group_names:
            self.group_names.append(groupname)
            self.group_embedsize.append(fconf["embedding_dimension"])
            self.group_vectors[groupname] = [tensor]
        else:
            self.group_vectors[groupname].append(tensor)

    def add_rawfeature(self, tensor, groupname, fname, fconf):
        self.feaconf[fname] = fconf
        self.id_names.append(fname)
        self.id_tensors.append(tensor)

        if groupname not in self.group_names:
            self.group_names.append(groupname)
            self.group_embedsize.append(1)
            self.group_vectors[groupname] = [tensor]
        else:
            self.group_vectors[groupname].append(tensor)

    def add_ptime(self, name, tensor, fconf):
        self.feaconf[name] = fconf
        self.pt_tensors.append(tensor)
        self.pt_names.append(name)

    def add_mask(self, name, tensor, fconf):
        # self.mask_tensors N * <L,1>
        self.feaconf[name] = fconf
        self.mask_tensors.append(tensor)
        self.mask_names.append(name)

    def get_vectorlist_bygroupname(self, groupname):
        if groupname not in self.group_vectors:
            print(self.group_vectors.keys())
            print(self.group_names)
        return self.group_vectors[groupname]

    def get_vector_groupasitem(self):
        '''
        return a list of list , L(seq len) * K(seq_fea_num) * < N(batch_size) * C(seq_fea_embedding_size)>
        inner list is of group's element ID's vector
        outter list is the sequence one by one
        '''
        elementlen = [len(x) for x in self.group_vectors.values()]
        if any([x != elementlen[0] for x in elementlen]):
            raise ValueError('element of group sequence lengths are not equal!')
        res = []
        for i in range(elementlen[0]):
            elementvecs = []
            for gname in self.group_names:
                vlist = self.group_vectors[gname]
                elementvecs.append(vlist[i])
            res.append(elementvecs)
        return res

    def get_ptimes(self, idx=None):
        if idx is None:
            return self.pt_tensors
        else:
            return self.pt_tensors[idx]

    def get_masks(self, idx=None):
        '''
        return L * <N,1>
        '''
        if idx is None:
            return self.mask_tensors
        else:
            return self.mask_tensors[idx]

    def get_embedsize(self, gname=None):
        if gname is None:
            return reduce(lambda x, y: x + y, self.group_embedsize)
        idx = -1
        for i in range(len(self.group_names)):
            if gname == self.group_names[i]:
                idx = i
                break
        return self.group_embedsize[idx]

    def get_embedsize_byname(self, fname=None):
        if "embedding_dimension" not in self.feaconf[fname]:
            return 1
        else:
            return self.feaconf[fname]["embedding_dimension"]


class EmbedFeature(object):
    def __init__(self, groupname, datalist):
        self.allsampledata = datalist
        self.groupname = groupname
        self.idxs = []
        self.fnames = []
        self.featuretensor = {}
        self.embedsize = {}
        self.feaconf = {}
        self.feaName2idx = {}
        self.idx2feaName = {}

    @property
    def embededsize(self):
        return self.get_embedsize()

    def add_idfeature(self, fname, idx_in_sample, embed_size, idx_in_matrix=None, fconf=None):
        self.featuretensor[fname] = self.allsampledata[idx_in_sample]
        self.idxs.append(idx_in_sample)
        self.fnames.append(fname)
        self.feaconf[fname] = fconf
        self.embedsize[fname] = embed_size
        self.feaName2idx[fname] = idx_in_sample
        self.idx2feaName[idx_in_sample] = fname

    def get_embedsize_byname(self, name):
        return self.embedsize[name]

    def get_embedsize(self):
        if len(self.embedsize.values()) == 0:
            return 0
        return reduce(lambda x, y: x + y, self.embedsize.values())

    def get_fgconf_byname(self, name):
        return self.feaconf[name]

    def get_vector_byname(self, name):
        return self.featuretensor[name]

    def get_names(self):
        return self.fnames

    def get_vectors(self, idxs=None, excludeFeaNameList=None):
        if idxs is None:
            idxs = self.idxs
        embedvectors = []
        exclude_idx = []
        if excludeFeaNameList is not None:
            exclude_idx = [self.feaName2idx[i] for i in excludeFeaNameList]
        for idx in idxs:
            if idx in exclude_idx:
                continue
            embedvectors.append(self.allsampledata[idx])
        return embedvectors


class RawFeature(object):
    def __init__(self, groupname, datalist, usebn=False):
        self.allsampledata = datalist
        self.groupname = groupname
        self.idxs = []
        self.fnames = []
        self.featuretensor = {}
        self.feaconf = {}
        self.feaName2idx = {}
        self.idx2feaName = {}

        self.raw2orders = {}
        self.raw3orders = {}
        self.raw4orders = {}

        self.BN_mean = {}
        self.BN_var = {}
        self.usebn = usebn
        self.bnedtensors = {}

    def add_rawfeature(self, fname, idx_in_sample, fconf=None):
        self.featuretensor[fname] = self.allsampledata[idx_in_sample]
        self.idxs.append(idx_in_sample)
        self.fnames.append(fname)
        self.feaconf[fname] = fconf
        self.feaName2idx[fname] = idx_in_sample
        self.idx2feaName[idx_in_sample] = fname

        tt = self.allsampledata[idx_in_sample]
        self.raw2orders[idx_in_sample] = tt * tt
        self.raw3orders[idx_in_sample] = tt * self.raw2orders[idx_in_sample]
        self.raw4orders[idx_in_sample] = tt * self.raw3orders[idx_in_sample]
        '''
        self.tensor = self.allsampledata[idx_in_sample]
        if self.usebn:
            mean, var = tf.nn.moments(self.tensor, axes=[0])
            self.BN_mean[idx_in_sample] = mean
            self.BN_var[idx_in_sample] = var
            offset = tf.Variable(tf.zeros([1]))
            scale = tf.Variable(tf.ones([1]))
            c = idx_in_sample
            batch_norm = tf.nn.batch_normalization(self.allsampledata[c], self.BN_mean[c], self.BN_var[c], offset, scale, 0.001)
            self.bnedtensors[idx_in_sample] = batch_norm
        '''

    def get_fgconf_byname(self, name):
        return self.feaconf[name]

    def get_tensor_byname(self, name):
        return self.featuretensor[name]

    def get_tensors(self, excludeFeaNameList=None, normStat=None):
        '''
        excludeFeaNameList: a list of exclude feature names
        normStat: is a dict like {feaName:{"mean":0,"std":1},...}
        '''
        self.tensor_ids = []
        exclude_idx = []
        if excludeFeaNameList is not None:
            exclude_idx = [self.feaName2idx[i] for i in excludeFeaNameList]
        for c in self.idxs:
            # batch_size = self.allsampledata[c].get_shape().as_list()[0]
            '''
            if self.usebn:
                self.tensor_ids.append(self.bnedtensors[c])
            else:
                self.tensor_ids.append(self.allsampledata[c])
            '''
            if c in exclude_idx:
                continue
            if normStat is not None and normStat.has_key(self.idx2feaName[c]):
                assert (normStat[self.idx2feaName[c]]["std"] != 0)
                self.tensor_ids.append(
                    (self.allsampledata[c] - normStat[self.idx2feaName[c]]["mean"]) / normStat[self.idx2feaName[c]][
                        "std"])
            else:
                self.tensor_ids.append(self.allsampledata[c])
        return self.tensor_ids

    def get_tensors_4orders(self):
        self.tensor_ids = []
        for c in self.idxs:
            # batch_size = self.allsampledata[c].get_shape().as_list()[0]
            '''
            if self.usebn:
                self.tensor_ids.append(self.bnedtensors[c])
            else:
                self.tensor_ids.append(self.allsampledata[c])
            '''
            self.tensor_ids.append(self.allsampledata[c])
        return self.tensor_ids

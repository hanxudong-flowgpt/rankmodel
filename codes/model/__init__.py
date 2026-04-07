from pdd import shareitem_cxr_model
from pdd import shareitem_cxr_esmm
from pdd import rank_cxr_model
from pdd import rank_cxr_bert_model
from pdd import rank_cxr_esmm


MODELZOO = {}
MODELZOO['shareitem_cxr_model'] = pdd.shareitem_cxr_model.KonduModel
MODELZOO['shareitem_cxr_model2'] = pdd.shareitem_cxr_model.KonduModel
MODELZOO['shareitem_cxr_model3'] = pdd.shareitem_cxr_model.KonduModel
MODELZOO['shareitem_cxr_esmm'] = pdd.shareitem_cxr_esmm.KonduModel
MODELZOO['rank_cxr_model'] = pdd.rank_cxr_model.KonduModel
MODELZOO['rank_cxr_esmm'] = pdd.rank_cxr_esmm.KonduModel
MODELZOO['rank_cxr_bert_model'] = pdd.rank_cxr_bert_model.KonduModel


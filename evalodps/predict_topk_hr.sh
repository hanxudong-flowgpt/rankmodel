#!/usr/bin/env bash

bizdate=20190527
yes2day=20190523

topkouttable='kd_online_ctr_top10query_top3k'
hrouttable='kd_online_top10query_top3k_ctrhr'


# sh pvmodel_predict_ivec.sh kd_hx_dram_fastpv_v2 kd_test_fastpv_v2ctr_ivec ${bizdate} search_algo_quality_dev kdquality
# sh pvmodel_predict_uvec.sh kd_hx_dram_fastpv_v2 kd_test_fastpv_v2ctr_uvec ${bizdate} ${yes2day} search_algo_quality_dev kdquality
# sh topk_hr.sh ${bizdate} ${yes2day} search_offline.kd_online_ctr_uvec_top10query search_offline.kd_online_ivec_top10query kd_online_ctr_top10query_top3k search_algo_quality_dev

ODPSCMD="odpscmd"

$ODPSCMD -e "
create table if not exists ${hrouttable}
(
    rn string,
    pos_num bigint,
    candi_num bigint,
    hit_num bigint
)
partitioned by (ds string)
lifecycle 15;

set odps.sql.mapper.split.size=512;

insert overwrite table ${hrouttable} partition(ds=${bizdate})
select
    tb.rn,
    char_matchcount(gt_nids,','),
    char_matchcount(vector,','),
    get_hit_num(gt_nids, vector)
from
(
    select get_pos_nid(label, 'ctr') as gt_nids, split_part(id, '_', 1) as rn from search_offline.lsc_log_wide_wdlfg_ctr_pv_recall where ds=${bizdate}
)ta
join
(
    select split_part(id, '_', 1) as rn, vector from search_offline.kd_online_ctr_top10query_top3k where ds=${bizdate}
)tb
on ta.rn=tb.rn
;
select avg(hit_num/pos_num) from ${hrouttable} where ds=${bizdate} and pos_num>0;"
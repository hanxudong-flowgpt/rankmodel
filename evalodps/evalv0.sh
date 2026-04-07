ivecdate='20181224'
ivectable='search_offline.kd_hx_baseline_ivec'
quvecdate='20181225'
quvectable='search_offline.kd_hx_baseline_uvec'
pooltable='search_offline.kd_hx_eval_queryitemrn_pool_v0'
#pooltable='search_offline_dev.kd_hx_baseline_hitrate_v0_pool'
jobname='kd_hx_baseline_hr'

ODPSCMD="/opt/taobao/tbdpapp/odpswrapper/odpswrapper.py"

$ODPSCMD -e "
create table if not exists ${jobname}_v0
(
    rn string,
    --page_id bigint,
    pos bigint,

    user_id string,
    query string,
    queryseg_norm string,
    queryseg_norm_encode string,
    item_id string,

    isclick bigint,
    ispay bigint,
    iscart bigint,
    --iscollect bigint,


    uvec string,
    ivec string,
    uivec_inner double
)
partitioned by(ds string)
lifecycle 30;

set odps.sql.mapper.split.size=4;
set odps.sql.reducer.instances=4000;
set odps.sql.joiner.instances=4000;
insert overwrite table ${jobname}_v0 partition(ds=$quvecdate)
select
    tt.rn,
    pos,

    tt.user_id,
    query,
    tt.queryseg_norm,
    tt.queryseg_norm_encode,
    tb.item_id,

    click,
    pay,
    add_cart,

    uvec,
    ivec,
    inner_product(uvec, ivec)
from
(
    select * from $pooltable where ds=$quvecdate
)tt
join
(
    select 
        vector as ivec,
        item_id
    from $ivectable where ds=$ivecdate
)tb
on tt.item_id=tb.item_id
join
(
    select 
        vector as uvec,
        rn,
        user_id,
        queryseg_norm_encode
    from $quvectable where ds=$quvecdate
)tbb
on tt.rn=tbb.rn
left outer join
(
    select
        rn,
        pos,

        userid,
        query,
        aucid as item_id,

        if(ipv>0, 1, 0) as click,
        pay,
        add_cart
    from search_offline.random_sample_pv_ipv_pay where ds=$quvecdate
)tc
on tc.rn=tt.rn and tc.item_id=tt.item_id
;
"

$ODPSCMD -e "
create table if not exists ${jobname}_v0_result 
(
    rn string,

    itempool_num bigint,
    exposure_num bigint,
    click_num bigint,

    top10hit bigint,
    top30hit bigint,
    top50hit bigint,
    top70hit bigint,
    top90hit bigint,
    top110hit bigint,
    top130hit bigint,
    top150hit bigint,
    top170hit bigint
)
partitioned by(ds string)
lifecycle 30;


insert overwrite table ${jobname}_v0_result partition(ds=$quvecdate)
select
    rn,
    count(1) as itempool_num,
    sum(if(pos is null, 0, 1)) as exposure_num,
    sum(if(isclick is null, 0, isclick)) as click_num,
    kd_get_top_k_hitrate(uivec_inner, if(isclick is null, 0, isclick), 10 ) as top10hit,
    kd_get_top_k_hitrate(uivec_inner, if(isclick is null, 0, isclick), 30 ) as top30hit,
    kd_get_top_k_hitrate(uivec_inner, if(isclick is null, 0, isclick), 50 ) as top50hit,
    kd_get_top_k_hitrate(uivec_inner, if(isclick is null, 0, isclick), 70 ) as top70hit,
    kd_get_top_k_hitrate(uivec_inner, if(isclick is null, 0, isclick), 90 ) as top90hit,
    kd_get_top_k_hitrate(uivec_inner, if(isclick is null, 0, isclick), 110 ) as top110hit,
    kd_get_top_k_hitrate(uivec_inner, if(isclick is null, 0, isclick), 130 ) as top130hit,
    kd_get_top_k_hitrate(uivec_inner, if(isclick is null, 0, isclick), 150 ) as top150hit,
    kd_get_top_k_hitrate(uivec_inner, if(isclick is null, 0, isclick), 170 ) as top170hit
from ${jobname}_v0 where ds=$quvecdate
group by rn;

select * from ${jobname}_v0_result where ds=$quvecdate limit 100;

select
    avg(cast(top10hit as double)/(click_num+1e-8)),
    avg(cast(top30hit as double)/(click_num+1e-8)),
    avg(cast(top50hit as double)/(click_num+1e-8)),
    avg(cast(top70hit as double)/(click_num+1e-8)),
    avg(cast(top90hit as double)/(click_num+1e-8)),
    avg(cast(top110hit as double)/(click_num+1e-8)),
    avg(cast(top130hit as double)/(click_num+1e-8)),
    avg(cast(top150hit as double)/(click_num+1e-8)),
    avg(cast(top170hit as double)/(click_num+1e-8))
from
    ${jobname}_v0_result
where ds=$quvecdate and click_num>0;
"
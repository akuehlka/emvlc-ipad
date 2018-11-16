select s.dataset, s.sequenceid, i.contacts, i.contacts_cosmetic, i.contacts_texture, a.contacts as an1, b.contacts as an2
from sample s
join images i on s.sequenceid=i.sequenceid
left join (
    select *
    from ndcld_ann_agree2
    where annotator='akuehlka'
    and agree_labels=0) a on s.sequenceid=a.sequenceid
left join (
    select *
    from ndcld_ann_agree2
    where annotator='aczajka'
    and agree_labels=0) b on s.sequenceid=b.sequenceid
where a.agree_labels=0
  and b.agree_labels=0;
  
select s.dataset, s.sequenceid, i.tag_list, a.contacts as an1, b.contacts as an2
from sample s
join images i on s.sequenceid=i.sequenceid
left join (
    select *
    from ndcld_ann_agree2
    where annotator='akuehlka'
    and agree_tags=0) a on s.sequenceid=a.sequenceid
left join (
    select *
    from ndcld_ann_agree2
    where annotator='aczajka'
    and agree_tags=0) b on s.sequenceid=b.sequenceid
where a.agree_tags=0
  and b.agree_tags=0;
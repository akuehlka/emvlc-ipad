select count(*)
from images
where sequenceid not in (select sequenceid from livdet)
  and sequenceid not in (select sequenceid from ndcld12)
  and sequenceid not in (select sequenceid from ndcld13)
  and sequenceid not in (select sequenceid from ndcld14)
  and sequenceid not in (select sequenceid from ndcld15);

select sequenceid, path 
from (  
    select sequenceid from livdet union
    select sequenceid from ndcld12 union
    select sequenceid from ndcld13 union
    select sequenceid from ndcld15
    group by sequenceid) as i
join filemap f on i.sequenceid=f.sequenceid
order by random()
limit 1200;

create table ndcld_annotation 
as 
select 'akuehlka' as annotator, sequenceid, contacts
from antmp;

delete from antmp;

insert into ndcld_annotation
select 'aczajka', sequenceid, contacts
from antmp;

select a.sequenceid, a.contacts, i.contacts, i.contacts_type, i.contacts_texture, i.contacts_toric, i.contacts_cosmetic,  i.tag_list like '%texture%' as tag_texture, 
i.tag_list like '%soft%' as tag_soft, 
i.tag_list like '%NoContacts%' as tag_nocontacts,
length(i.tag_list) = 0 as tag_empty
from ndcld_annotation a
join images i on a.sequenceid=i.sequenceid;

select tag_list
from images
where sequenceid='06130d1292';

drop table sample;

create table sample
as select 'livdet' dataset, sequenceid from vlivdet;

insert into sample
select 'ndcld12'dataset, sequenceid from vndcld12;
insert into sample
select 'ndcld13'dataset, sequenceid from vndcld13;
insert into sample
select 'ndcld15'dataset, sequenceid from vndcld15;

select s.sequenceid, f.path
from sample s
join filemap f on s.sequenceid=f.sequenceid
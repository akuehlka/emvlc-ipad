files = dir('osiris_output/*.txt');

result = {};

for i=1:size(files,1)
    fprintf('%s\n',files(i).name);
    fname = files(i).name;
    sequenceid = regexprep(fname,'_para.txt','');
    [xp,yp,rp,xi,yi,ri] = irisCircApprox_OSIRISsegm(['osiris_output/' fname]);
    result = cat(1, result, [{sequenceid}, xp, yp, rp, xi, yi, ri]);
end

t = table(result(:,1),result(:,2),result(:,3),result(:,4),result(:,5),result(:,6),result(:,7));
t.Properties.VariableNames = {'sequenceid','xp','yp','rp','xi','yi','ri'};
writetable(t,'osiris_coords.csv');
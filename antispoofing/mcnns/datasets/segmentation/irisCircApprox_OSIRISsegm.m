function [xp,yp,rp,xi,yi,ri] = irisCircApprox_OSIRISsegm(filename)

c = dlmread(filename);
NoOfPupilPoints = c(1);
NoOfIrisPoints = c(2);

PupilPoints = c(3,1:3*NoOfPupilPoints);
IrisPoints = c(4,1:3*NoOfIrisPoints);

[xp,yp,rp] = circfit(PupilPoints(1:3:end),PupilPoints(2:3:end));
[xi,yi,ri] = circfit(IrisPoints(1:3:end),IrisPoints(2:3:end));

function [xc,yc,r] = circfit(x,y)
x = x(:); 
y = y(:);
a = [x y ones(size(x))]\[-(x.^2+y.^2)];
xc = -.5*a(1);
yc = -.5*a(2);
r = sqrt((a(1)^2+a(2)^2)/4-a(3));
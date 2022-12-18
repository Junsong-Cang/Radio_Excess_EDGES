cd /Users/cangtao/IHEPBox/Projects/GitHub/Radio_Excess_EDGES/Example_emcee
clear

a=2
w=3
t=0.5

clc
LineWidth=2;
PlotSize=25;
x=0:0.1:pi;
y=a*sin(w*x+t);
y=a*x+w;

F='/Users/cangtao/IHEPBox/Projects/GitHub/Radio_Excess_EDGES/Example_emcee/Fake_Data_Sine.txt';
L=length(x);
delete(F)
fid=fopen(F,'w');
for idx=1:L
    fprintf(fid,'%f    %f\n',x(idx), y(idx));
end
fclose(fid);

clf
plot(x,y,'k','LineWidth',LineWidth);hold on
xlabel('$x$','Interpreter','latex','FontSize',PlotSize,'FontName','Times');
ylabel('$y$','Interpreter','latex','FontSize',PlotSize,'FontName','Times');
set(gca,'FontSize',PlotSize,'Fontname','Times');


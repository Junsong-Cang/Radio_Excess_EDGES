cd /Users/cangtao/IHEPBox/Projects/GitHub/Radio_Excess_EDGES
clear

LineWidth=2
PlotSize=25
Error=0.1

clc

D='/Users/cangtao/IHEPBox/Projects/GitHub/Radio_Excess_EDGES/Data_EDGES_v0.txt';
F='/Users/cangtao/IHEPBox/Projects/GitHub/Radio_Excess_EDGES/Data_EDGES_2.txt';
D=importdata(D);
D=D.data;

f=D(:,1);
z=1420./f-1;
D(:,end+1)=Error;
S=size(D);

T21_model=D(:,end-2);
T21_signal=D(:,end-1);

delete(F)
fid=fopen(F,'w');
for Idx=1:S(1)
    fprintf(fid,'%f    %i    %E    %E    %E    %E    %E\n',D(Idx,1), D(Idx,2), D(Idx,3), D(Idx,4), D(Idx,5), D(Idx,6), D(Idx,7));
end
fclose(fid);

clf
plot(z,T21_model,'k','LineWidth',LineWidth);hold on
plot(z,T21_signal,'r','LineWidth',LineWidth);hold on
xlabel('$z$','Interpreter','latex','FontSize',PlotSize,'FontName','Times');
ylabel('$T_{21}$ (k)','Interpreter','latex','FontSize',PlotSize,'FontName','Times');
set(gca,'FontSize',PlotSize,'Fontname','Times');
title('EDGES Signal','Interpreter','latex','FontSize',PlotSize)
axis([-Inf Inf -0.6 0.1]);
LgD=legend('Best-Fit',...
    'Signal');
set(LgD,'Interpreter','latex','Location','SouthEast','FontSize',PlotSize)
% TICKS=10.^[-30:30];
% xticks([TICKS])
% yticks([TICKS])



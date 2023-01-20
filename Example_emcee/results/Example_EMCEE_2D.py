import getdist.plots as plots, os
g=plots.GetDistPlotter(plot_data=r'./plot_data')
g.settings.setWithSubplotSize(4.0)
roots = ['Example_EMCEE']
pairs=[]
pairs.append(['p1','p2'])
pairs.append(['p1','p3'])
pairs.append(['p2','p3'])
g.plots_2d(roots,param_pairs=pairs,filled=True)
g.export(os.path.join(r'/Users/cangtao/IHEPBox/Projects/GitHub/Radio_Excess_EDGES/Example_emcee/results',r'Example_EMCEE_2D.eps'))

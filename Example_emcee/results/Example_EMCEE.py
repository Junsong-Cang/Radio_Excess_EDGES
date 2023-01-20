import getdist.plots as plots, os
g=plots.GetDistPlotter(plot_data=r'./plot_data')
g.settings.setWithSubplotSize(4.0)
roots = ['Example_EMCEE']
markers={}
g.plots_1d(roots, markers=markers)
g.export(os.path.join(r'/Users/cangtao/IHEPBox/Projects/GitHub/Radio_Excess_EDGES/Example_emcee/results',r'Example_EMCEE.eps'))

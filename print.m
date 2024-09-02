%% Print figure to file using input name.

function print(name)

set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex'); 
set(0,'defaultLegendInterpreter','latex');
set(gcf,'color','w')
savefig(name)
exportgraphics(gcf,[name,'.pdf'])

end
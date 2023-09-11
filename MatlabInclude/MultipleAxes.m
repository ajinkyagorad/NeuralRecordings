% Create some plotting data and plot
x = 0:0.1:2*pi;   y = sin(x);
% Plot, can specify line attributes (like LineWidth) either 
% - inline: plot(x,y,'linewidth',2)
% - after: p1 = plot(x,y); p1.LineWidth = 2;
plot(x,y);
% Get current axes object (just plotted on) and its position
ax1 = gca;
axPos = ax1.Position;
% Change the position of ax1 to make room for extra axes
% format is [left bottom width height], so moving up and making shorter here...
ax1.Position = axPos + [0 0.16 0 -0.3];
% Exactly the same as for plots (above), axes LineWidth can be changed inline or after
ax1.LineWidth = 2;
% Add two more axes objects, with small multiplier for height, and offset for bottom
ax2 = axes('position', (axPos .* [1 1 1 1e-3]) + [0 0.08 0 0], 'color', 'none', 'linewidth', 2);
ax3 = axes('position', (axPos .* [1 1 1 1e-3]) + [0 0.00 0 0], 'color', 'none', 'linewidth', 2);
% You can change the limits of the new axes using XLim
ax2.XLim = [0 10];
ax3.XLim = [100 157];
% You can label the axes using XLabel.String
ax1.XLabel.String = 'Lambda [nm]';
ax2.XLabel.String = 'Velocity [m/s]';
ax3.XLabel.String = 'Energy [eV]';
set(gcf,'Color','w');set(findobj(gcf,'type','axes'),'FontName','Consolas','FontSize',8,'FontWeight','Bold', 'LineWidth', 1);

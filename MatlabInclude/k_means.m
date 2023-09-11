% cluster according to k means algorithm
% v(matrix) has the elements as column vectors, to be grouped in k clusters
function [id,c]=k_means(v,k,show)
if(nargin<3) show = 0;end;
[p,N] = size(v); % number of attributes(p)  and sample points(N)
id = ones(1,N); % assigned cluster id
%%
if(show) % if to display the algorithm
    ColorRef  = distinguishable_colors(k,'w');
    if(p==1)
        h = plot(v(1,:),'.');
    elseif(p==2)
        h = plot(v(1,:),v(2,:),'.');
    elseif(p==3)
        h =scatter3(v(1,:),v(2,:),v(3,:),'.');
    else
        vtemp=pca(v)';
        h =scatter3(vtemp(1,:),vtemp(2,:),vtemp(3,:),'.');
    end
end
col='w';set(gcf,'Color',col);set(gca, 'Color',col);set(findobj(gcf, 'Type', 'Legend'),'Color',col);
%% Start
loop = 1;
old_id = id;
c = v(:,randi([1 N],1,k));%centroid of the data points take first k (must be careful not to choose points too close)
while(loop)
    % get distance for current set of points
    
    distance2 = 0;
    for i = 1:p
        distance2 = distance2+(v(i,:)-c(i,:)').^2;
    end
    % find centroid for minimum distance
    [~,id] = min(distance2,[],1);
    if(id==old_id)
        loop = 0;
    end
    old_id = id;
    
    % update for new centroids
    for i = 1:k
        c(:,i) = mean(v(:,id==i),2);
    end
    % display progress in plot
    if(show)
        Color = zeros(N,3);
        for i = 1:k
            Color(id==i,:) = ColorRef(i,:).*ones(length(find(id==i)),1);
        end
        set(h,'CData',Color);
        drawnow;
    end
end

end

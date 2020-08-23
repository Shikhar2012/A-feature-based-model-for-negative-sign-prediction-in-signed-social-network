clc
close all
clear all
data = textread('epi.txt');
directed = false;
%% Load data
% disp('Loading data')
% load(dataFile)
[n,~,tMax] = size(data   );
%% Compute mean number of edges and edge probability
node_existing = false(n   ,n,tMax-1);
% cummulative_adj = zeros(n);
% for i = 1:tMax-1
%     cummulative_adj = cummulative_adj|adj(:,:,i);    % All the edges that have formed until time 'i'
%     node_existing(:,:,i) = cummulative_adj;
% end
node_new = ~node_existing;
 
nodeActive = isNodeActive(data(:,3));
nodeActive = cumsum(nodeActive,2);
nodeActive(nodeActive > 0) = 1;

% activeMask = false(n,n,tMax);
% for t = 1:tMax
%     activeMaskCurr = nodeActive(:,t)*nodeActive(:,t)';
%     activeMaskCurr(diag(true(n,1))) = 0;
%     activeMask(:,:,t) = activeMaskCurr;
%     if directed == false
%         activeMask(:,:,t) = tril(activeMask(:,:,t));
%     end
% end

nNodePairs = zeros(1,tMax);
nEdges = zeros(1,tMax);
for t = 1:tMax
    nNodePairs(t) = nnz(data(:,3));
    nEdges(t) = nnz(data(:,3));
    if directed == false
        nEdges(t) = nEdges(t)/2;
    end
end

fprintf('Total number of nodes: %i\n', nnz(nodeActive(:,end)))
fprintf('Mean number of edges: %f\n', mean(nEdges))
fprintf('Mean edge probability: %f\n', mean(nEdges./nNodePairs))

%% Compute mean new and previous edge probabilities
% data=nodeActive;
%% Train 
% jj=getJaccard(n,nNodePairs);
% rank (nodeActive);
G = graph(n,nNodePairs);
figure,plot(G);
N = neighbors(G,1);
D = degree(G);
eid = outedges(G,1);
P = shortestpath(G,1,807);
nn = nearest(G,1,807);
G.Edges(eid,:);
kk=getCosineSimilarity(D,nodeActive);
similarity = dice(nodeActive,D);
pg_ranks = centrality(G,'pagerank');
coeff = 0.9920;
dataclass = data(:,1:2);
classtrain = data(:,3);
cl = fitcsvm(dataclass,classtrain,'KernelFunction','rbf','BoxConstraint',Inf,'ClassNames',[-1,1]);
d = 0.01;
[x1Grid,x2Grid] = meshgrid(min(dataclass(:,1)):d:max(dataclass(:,1)),min(dataclass(:,2)):d:round(coeff(1)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(cl,xGrid);
round(times(cl.Prior(2),10))
% Plot the data and the decision boundary
figure;
h(1:2) = gscatter(dataclass(:,1),dataclass(:,2),classtrain,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(dataclass(cl.IsSupportVector,1),dataclass(cl.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
hold off
%----------------
cl2 = fitcsvm(dataclass,classtrain,'KernelFunction','rbf');
[~,scores2] = predict(cl2,xGrid);
round(times(cl2.Prior(2),10))
figure;
h(1:2) = gscatter(dataclass(:,1),dataclass(:,2),classtrain,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(dataclass(cl2.IsSupportVector,1),dataclass(cl2.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores2(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
hold off
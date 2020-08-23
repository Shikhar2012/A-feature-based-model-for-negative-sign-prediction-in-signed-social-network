clc
close all
clear all
dataFile = 'wiki.mat';
directed = false;
%% Load data
disp('Loading data')
load(dataFile)
[n,~,tMax] = size(data);
%% Compute mean number of edges and edge probability
node_existing = false(n   ,n,tMax-1);
cummulative_adj = zeros(n);
for i = 1:tMax-1
    cummulative_adj = cummulative_adj|adj(:,:,i);    % All the edges that have formed until time 'i'
    node_existing(:,:,i) = cummulative_adj;
end
node_new = ~node_existing;
 
nodeActive = isNodeActive(data);
nodeActive = cumsum(nodeActive,2);
nodeActive(nodeActive > 0) = 1;

activeMask = false(n,n,tMax);
for t = 1:tMax
    activeMaskCurr = nodeActive(:,t)*nodeActive(:,t)';
    activeMaskCurr(diag(true(n,1))) = 0;
    activeMask(:,:,t) = activeMaskCurr;
    if directed == false
        activeMask(:,:,t) = tril(activeMask(:,:,t));
    end
end

nNodePairs = zeros(1,tMax);
nEdges = zeros(1,tMax);
for t = 1:tMax
    nNodePairs(t) = nnz(activeMask(:,:,t));
    nEdges(t) = nnz(data(:,:,t));
    if directed == false
        nEdges(t) = nEdges(t)/2;
    end
end

fprintf('Total number of nodes: %i\n', nnz(nodeActive(:,end)))
fprintf('Mean number of edges: %f\n', mean(nEdges))
fprintf('Mean edge probability: %f\n', mean(nEdges./nNodePairs))

%% Compute mean new and previous edge probabilities
adj = data(:,:,2:end);
newDens = zeros(1,tMax-1);
existingDens = zeros(1,tMax-1);

activeMask = activeMask(:,:,1:tMax-1);
activeMaskNew = node_new & activeMask;
activeMaskExisting = node_existing & activeMask;

for t = 1:tMax-1
    adjCurr = adj(:,:,t);
    newDens(t) = nnz(adjCurr(activeMaskNew(:,:,t)))/nnz(activeMaskNew(:,:,t));
    existingDens(t) = nnz(adjCurr(activeMaskExisting(:,:,t))) ...
        / nnz(activeMaskExisting(:,:,t));
end

% fprintf('Mean new edge probability: %f\n', mean(newDens))
% fprintf('Mean re-occurring edge probability: %f\n', mean(existingDens))


% G = graph(edges(:,1), edges(:,2));      % create a graph from edges
% G.Nodes = table(name);                  % name the nodes
% figure                                  % visualize the graph
% plot(G);
 
% plot(nNodePairs);      % create adjacency matrix
 A = full(activeMaskNew);            % convert the sparse matrix to full matrix
% plot(nodeActive);
plot(nNodePairs);
iter=2;
% data=nodeActive;
%% Train 
test=data(2501:3000,:); % Test Data
test_L=label(2501:3000,:); % Real Labels of Test Data
T2=test_L';
p2=test';
Net=newp([ones(1,3000)*(-1);ones(1,3000)]',size(label,2));
Net.TrainParam.epochs=iter; 
for i=1:10
    L=randsrc(1,1,50:2500);
    S=randperm(3000,L);
    p=data(S,:); % Train Data
    p=p';
    T=label(S,:); % Target Data (Labels)
    T=T';
    Net = feedforwardnet(100);
    Net=train(Net,p,T);
     Y = sim(Net,p); 
    Train_accuracy=sum(sum(Y==T))/(size(label,2)*length(S));
    YY2(i,:,:) = sim(Net,p2); % Result Labels for Test Data
end

for i=1:size(label,2)
    for j=1:500
        [~,index]=max([sum(YY2(:,i,j)==0),sum(YY2(:,i,j)==1)]);
        Y2(i,j)=index-1;
    end
end
Tt2=T2;
Yy2=Y2;
for jj=1:size(label,2)
    T2=Tt2(jj,:);
    Y2=Yy2(jj,:);
    TP=0; TN=0; FN=0; FP=0;
    for i=1:size(T2,1)*size(T2,2)
        if(Y2(i)==T2(i) && Y2(i)==1)
            TP=TP+1;
        elseif(Y2(i)==T2(i) && Y2(i)==0)
            TN=TN+1;
        elseif(Y2(i)~=T2(i) && Y2(i)==1)
            FP=FP+1;
        elseif(Y2(i)~=T2(i) && Y2(i)==0)
            FN=FN+1;
        end
    end
    %% Results
    accuracy=(TP+TN)/(TP+TN+FP+FN);
    precision= TP/ (TP + FP);
    recall= TP/ (TP + FN);
    acc(jj)=accuracy;
    precision= TP/ (TP + FP);
    if isnan(precision)
        precision=0;
    end
    pre(jj)=precision;
    recall= TP/ (TP + FN);
    if isnan(recall)
        recall=0;
    end
    rec(jj)=recall;
    F_measure=(2*precision*recall)/(precision+recall);
    if isnan(F_measure)
        F_measure=0;
    end
    Fmea(jj)=F_measure;
end
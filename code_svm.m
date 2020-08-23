clc
close all
clear all
data= textread('slash1.txt');
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
 
nodeActive = isNodeActive(data(:,2));
nodeActive = cumsum(nodeActive,2);
nodeActive(nodeActive > 0) = 1;
nNodePairs = zeros(1,tMax);
nEdges = zeros(1,tMax);
for t = 1:tMax
    nNodePairs(t) = nnz(data(:,2));
    nEdges(t) = nnz(data(:,2));
    if directed == false
        nEdges(t) = nEdges(t)/2;
    end
end

fprintf('Total number of nodes: %i\n', nnz(nodeActive(:,end)))
fprintf('Mean number of edges: %f\n', mean(nEdges))
fprintf('Mean edge probability: %f\n', mean(nEdges./nNodePairs))
G = graph(n,nNodePairs);
N = neighbors(G,1);
D = degree(G);
eid = outedges(G,1317);
% P = shortestpath(G,1,807);
nn = nearest(G,1,807);
G.Edges(eid,:);
kk=getCosineSimilarity(D,nodeActive);
similarity = dice(nodeActive,D);
pg_ranks = centrality(G,'pagerank');
s = n;
H = kk;
for c = 1:1
for r = 1:n
H(r,c) = 1/(r+c-1);
end
end
s = n;
H1 = nEdges;
for c = 1:1
for r = 1:n
H1(r,c) = 1/(r+c-1);
end
end
s = n;
H2 = similarity;
for c = 1:1
for r = 1:n
H2(r,c) = 1/(r+c-1);
end
end
s = n;
H3 = eid;
for c = 1:1
for r = 1:n
H3(r,c) = 1/(r+c-1);
end
end
feature=[D nodeActive pg_ranks H H1 H2 H3];
 Problem.obj = @(feature) Sphere(feature);
%  Problem.obj = sum(feature.^4 + feature.^3 + feature.^2 + feature.^1 + 1)
% Problem.obj= data(:,3).^2 - 1;
% data1=D;
% Problem.obj = sum(.^2));
Problem.nVar = 1317;

M = 20; % number of chromosomes (cadinate solutions)
N = Problem.nVar;  % number of genes (variables)
MaxGen = 100;
Pc = 0.85;
Pm = 0.01;
Er = 0.05;
visualization = 1; % set to 0 if you do not want the convergence curve 
[BestChrom]  = GeneticAlgorithm (M , N, MaxGen , Pc, Pm , Er , Problem.obj , visualization)
dataclass = data(:,1:2);
classtrain = data(:,2);
classtrain=uint8(classtrain);
class2=logical(classtrain);
mdlSVM = fitcsvm(BestChrom.Gene',class2,'Standardize',true);
mdlSVM = fitPosterior(mdlSVM);
[~,score_svm] = resubPredict(mdlSVM);
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(class2,score_svm(:,mdlSVM.ClassNames),'true');
% mdlNB = fitcnb(score_svm,scores);
% [~,score_nb] = resubPredict(mdlNB)   
% acc=max(score_nb(:,2));
o=bsxfun(@eq,feature,M);
o1=BestChrom.Gene'+nodeActive+o(:,3)+o(:,1);
PERFORMANCE_SVM=Evaluate(mdlSVM.IsSupportVector,o1)



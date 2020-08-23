function nodeActive = isNodeActive(adj)


[n,~,tMax] = size(adj);


nodeActive = false(n,tMax);
parfor t = 1:tMax
    nodeActive(:,t) = (sum(adj(:,:,t)) > 0) | (sum(adj(:,:,t),2) > 0)';
end

end


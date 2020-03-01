function [A,W,mu,b2,total_iter] = SBL_Multimodal_TD_DL(Y,p,T)
nS = length(Y);
s2 = p.s2_initial;
if(p.TD)
    b2 = p.b2_initial;
    Classes = size(T,1);
else
    b2 = zeros(nS,1);
end

N = size(Y{1},2);
unique_groups = unique(p.groups{1});

if p.display, 
    DQ = zeros(p.numIter,nS);
end

for k = 1:nS
    B{k} = zeros(p.n(k),length(unique_groups));
    for g = 1:length(unique_groups)
        B{k}(p.groups{k} == unique_groups(g),g) = 1;
    end
end

nG = zeros(length(unique_groups),1);
for g = 1:length(unique_groups)
    for k = 1:nS
        nG(g) = nG(g) + sum(p.groups{k} == unique_groups(g));
    end
end

for k = 1:nS
    for g = 1:length(unique_groups)
        B_back{k}(g,p.groups{k} == unique_groups(g)) = 1/(nG(g));
    end
end

gamma = ones(length(unique_groups),N);
numBatches = floor(N/p.batchSize);
ind = randperm(N);

if(p.TD), 
    T = T(:,ind); 
end

for k = 1:nS,
    Y{k} = Y{k}(:,ind);
end

ind_init = randperm(N,max(p.n));
for k = 1:nS
    d(k) = size(Y{k},1);
    A{k} = NormalizeMatrix(Y{k}(:,ind_init(1:p.n(k))));
    
    if(p.TD)
        W{k} = ones(size(T,1),p.n(k));
        I{k} = eye(d(k)+Classes);
    else
        I{k} = eye(d(k));
        W{k} = 0;
    end
    
    if(~p.SS), 
        used_ind = [];
    end
    
    for n = 1:numBatches, 
        Ybatch{n,k} = Y{k}(:,(n-1)*p.batchSize+1:n*p.batchSize);
    end
    Ybatch{end,k} = [Ybatch{end,k} , Y{k}(:,numBatches*p.batchSize+1:end)];
end

if(p.TD)
    for n = 1:numBatches, 
        Tbatch{n} = T(:,(n-1)*p.batchSize+1:n*p.batchSize); 
    end
    Tbatch{end} = [Tbatch{end} , T(:,numBatches*p.batchSize+1:end)];
end

total_iter = 0;

for k = 1:nS
    mu{k} = pinv(A{k})*Y{k};
    if ~p.SufficientStatistics
        if p.DA, 
            Sigma{k} = zeros(p.n(k),N);
        else
            Sigma{k} = zeros(p.n(k),p.n(k),N);
        end
    end
end

for outer_iter = 1:p.numIter
    if ~p.SS
        if p.SufficientStatistics
            [A,gamma,used_ind] = RemoveRedundancySBL(Yb,A,mu,gamma,used_ind);
        else
            [A,gamma,used_ind] = RemoveRedundancySBL(Y,A,mu,gamma,used_ind);
        end
    else
        [A,gamma] = RemoveRedundancySBL_SS(Y,A,mu,gamma,p);
    end
    
    for iter = 1:p.numIterInner
        if p.TD, 
            W_old = W; 
        end
        A_old = A;
        
        for k = 1:nS, 
            Yb{k} = Ybatch{mod(iter-1,numBatches)+1,k};
        end
        
        if p.TD,
            Tb = Tbatch{mod(iter-1,numBatches)+1}; 
        else
            Tb = 0;
        end
        
        Nbatch = size(Yb{1},2);
        Ninit = mod(iter-1,numBatches)*p.batchSize;
        gamma_batch = gamma(:,Ninit+1:Ninit+Nbatch);
        
        %Calculate sufficient statistics
        for k = 1:nS
            if p.SufficientStatistics, 
                [Sigma{k},mu{k}] = ComputeSufficientStatistics(Yb{k},Tb,A{k},W{k},gamma_batch,s2(k),b2(k),I{k},B{k},p);
            else
                if p.DA, 
                    [Sigma{k}(:,Ninit+1:Ninit+Nbatch),mu{k}(:,Ninit+1:Ninit+Nbatch)] = ComputeSufficientStatistics(Yb{k},Tb,A{k},W{k},gamma_batch,s2(k),b2(k),I{k},B{k},p);
                else
                    [Sigma{k}(:,:,Ninit+1:Ninit+Nbatch),mu{k}(:,Ninit+1:Ninit+Nbatch)] = ComputeSufficientStatistics(Yb{k},Tb,A{k},W{k},gamma_batch,s2(k),b2(k),I{k},B{k},p); 
                end
            end
        end
        
        if p.SufficientStatistics,
            offset = 0; 
        else
            offset = Ninit;
        end
        
        for n = 1:Nbatch
            gamma(:,n+Ninit) = 0;
            for k = 1:nS
                if p.DA, 
                    gamma(:,n+Ninit) = gamma(:,n+Ninit)+B_back{k}*(Sigma{k}(:,n+offset)+mu{k}(:,n+offset).^2);
                else
                    gamma(:,n+Ninit) = gamma(:,n+Ninit)+B_back{k}*(diag(Sigma{k}(:,:,n+offset))+mu{k}(:,n+offset).^2);
                end
            end
        end
        
        parfor k = 1:nS
            if p.DA,
                C{k} = diag(sum(Sigma{k},2))+mu{k}*mu{k}';
            else
                C{k} = sum(Sigma{k},3)+mu{k}*mu{k}'; 
            end
            
            if p.SufficientStatistics,
                YXt = Yb{k}*mu{k}'; 
            else
                YXt = Y{k}*mu{k}';
            end
            A{k} = NormalizeMatrix(YXt/C{k});
        end
        
        if(p.TD)
            parfor k = 1:nS
                if p.SufficientStatistics,
                    TXt = Tb*mu{k}';
                else
                    TXt = T*mu{k}'; 
                end
                W{k} = (TXt/C{k});
            end
        end
        
        if p.display == 1
            cols = (nS+p.TD);
            rows = 1;
            %             for k = 1:nS
            %                 DQ(iter,k) = mean(DictionaryQuality(p.A{k},A{k}) > 0.99);
            %                 subplot(rows,cols,k); plot(DQ(1:iter,k)); title(['s2 = ' num2str(s2(k)) ', iter = ' num2str(iter)]);
            %             end
            
            if(p.TD)
                paraml1.lambda = 0.5;
                paraml1.mode = 2;
                paraml1.iter = 1000;
                for k = 1:nS
                    Xval{k} = mexLasso(p.Yval{k},A{k},paraml1);
                    cRate(iter+(outer_iter-1)*p.numIterInner,k) = ClassificationError(p.Tval,W{k},Xval{k});
                    cRateTrain(iter+(outer_iter-1)*p.numIterInner,k) = ClassificationError(T,W{k},mu{k});
                    
                    subplot(10+nS,1,10+k); hold off
                    plot(cRate(1:iter+(outer_iter-1)*p.numIterInner,k)); hold all;
                    plot(cRateTrain(1:iter+(outer_iter-1)*p.numIterInner,k));
                    legend('Val','Train','Location','northwest');
                    title(['classification rate ' num2str(cRate(iter+(outer_iter-1)*p.numIterInner,k))])
                    %                     subplot(nS*2,1,(k-1)*nS+2);% hold off; plot(T(:,1)); hold all;
                    %                     plot(mu{k}(:,1));% plot(W{1}*mu{1}(:,1));
                    %                     title('Text');
                end
                
                z = ClassSpecificClassificationError(p.Tval,W{1},Xval{1});
                for class = 1:Classes,
                    subplot(Classes+nS,1,class);
                    ind = find(T(class,:) == 1);
                    plot(mean(abs(mu{1}(:,ind)),2));
                    
                    Wnormalized = NormalizeMatrixRows(W{1});
                    w1 = Wnormalized(class,:);
                    title([num2str(z(class)) ', ' num2str(max(abs(w1*Wnormalized(setdiff([1:Classes],class),:)')))]);
                end
                
                %                 subplot(4,1,1); hold off
                %                 plot(cRate(1:iter+(outer_iter-1)*p.numIterInner,1)); hold all;
                %                 plot(cRateTrain(1:iter+(outer_iter-1)*p.numIterInner,1));
                %                 legend('Val','Train','Location','Best');
                %                 title(['Text classification rate ' num2str(cRate(iter+(outer_iter-1)*p.numIterInner,1))])
                %                 subplot(4,1,2);% hold off; plot(T(:,1)); hold all;
                %                 plot(mu{1}(:,1));% plot(W{1}*mu{1}(:,1));
                %                 title('Text');
                
                %                 subplot(4,1,3); hold off
                %                 plot(cRate(1:iter+(outer_iter-1)*p.numIterInner,2)); hold all;
                %                 plot(cRateTrain(1:iter+(outer_iter-1)*p.numIterInner,2));
                %                 legend('Val','Train','Location','Best');
                %                 title(['Image classification rate ' num2str(cRate(iter+(outer_iter-1)*p.numIterInner,2))])
                %                 subplot(4,1,4);% hold off; plot(T(:,1)); hold all;
                %                 plot(mu{2}(:,1)); %plot(W{2}*mu{2}(:,1));
                %                 title('Image');
                
                drawnow;
                %                 plot(cRate(1:iter)); title(['Classification Rate, b2 = ' num2str(b2(1))]);
            else
                numPlots = 3;
                for k = 1:nS
                    tmp = DictionaryQuality(p.A{k},A{k});
                    DQ(iter,k) = mean(tmp > 0.99);
                    subplot(numPlots,nS,k); plot(DQ(1:iter,k)); title(['s2 = ' num2str(s2(k)) ', iter = ' num2str(iter)]);
                    dLds_hist(k,iter) = sum(dLds{k}(:,1));
                    subplot(numPlots,nS,k+nS); plot(dLds_hist(k,1:iter)); title(['History of derivative, s2 = ' num2str(s2(k))]);
                    subplot(numPlots,nS,k+nS*2); plot(tmp); title(['Dictionary quality']);
                    %                 subplot(numPlots,nS,k+nS*3); plot(mu{k}(:,1)); title('mu(:,1)');
                    %                 subplot(numPlots,nS,k+nS*4); plot(mean(mu{k}.^2,2)); title('Average mu');
                end
                drawnow;
            end
            %             subplot (rows,cols,1); hold off
            %             plot(p.Tval(:,1));
            %             hold all; plot(W{1}*Xval{1}(:,1)); drawnow;
            
            
        end
        
        for k = 1:nS
            update(k,1) = max(max(abs(A_old{k}-A{k})./abs(A_old{k})));
            
            if p.TD,
                update(nS+k,1) = max(max(abs(W_old{k}-W{k})./abs(W_old{k}))); 
            end
        end
        
        if (sum(update < 1e-3) == (nS+nS*p.TD)) && (mod(iter,numBatches) == 0),
            break; 
        end
        
        %Shuffle data
        if mod(iter,numBatches) == 1
            ind = randperm(N);
            gamma = gamma(:,ind);
            if ~p.SS, 
                parfor ii = 1:length(used_ind), 
                    used_ind(ii) = find(ind == used_ind(ii)); 
                end; 
            end
            
            for k = 1:nS
                Y{k} = Y{k}(:,ind);
                
                parfor n = 1:numBatches, 
                    Ybatch{n,k} = Y{k}(:,(n-1)*p.batchSize+1:n*p.batchSize);
                end
                Ybatch{end,k} = [Ybatch{end,k} , Y{k}(:,numBatches*p.batchSize+1:end)];
            end
            
            if p.TD
                T = T(:,ind);
                parfor n = 1:numBatches,
                    Tbatch{n} = T(:,(n-1)*p.batchSize+1:n*p.batchSize); 
                end
                Tbatch{end} = [Tbatch{end} , T(:,numBatches*p.batchSize+1:end)];
            end
            
            if ~p.SufficientStatistics
                if p.DA,
                    parfor k = 1:nS,
                        mu{k} = mu{k}(:,ind);
                        Sigma{k} = Sigma{k}(:,ind);
                    end
                else
                    parfor k = 1:nS, 
                        mu{k} = mu{k}(:,ind); 
                        Sigma{k} = Sigma{k}(:,:,ind); 
                    end
                end
            end
        end
    end
    
    %%Plot classification
    if False %outer_iter == p.numIter && 0
        for k = 1:nS
            tmpY = cell(1,1); tmpY{1} = p.Yval{k};
            
            tmpp.s2_initial = p.s2_initial(k);
            tmpp.s2_lowerbound = s2(k);
            tmpp.numIter = p.numIter;
            tmpp.numIterInner = p.numIterInner;
            tmpp.s2_decay_factor = exp(log(tmpp.s2_lowerbound./tmpp.s2_initial)/(outer_iter));
            tmpp.DA = p.DA;
            
            if k == 1
                tmpp.groups{1} = p.groups{1};
            else
                tmpp.groups{1} = p.groups{2};
            end
            
            tmpA = cell(1,1); tmpA{1} = A{k};
            Xval{k} = SBL_Multimodal(tmpY,tmpA,tmpp);
            cRate(outer_iter,k) = ClassificationError(p.Tval,W{k},Xval{k}{1});
            
            subplot(nS,1,k); plot(cRate(:,k)); drawnow;
        end
    end
    
    total_iter = total_iter+iter;
    
    parfor k = 1:nS
        s2(k) = max(s2(k)*p.s2_decay_factor(k),p.s2_lowerbound(k));
        if p.TD, 
            b2(k) = max(b2(k)*p.b2_decay_factor(k),p.b2_lowerbound(k));
        end
    end
end
end
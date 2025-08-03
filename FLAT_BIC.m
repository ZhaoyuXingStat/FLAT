function[beta_admm,BIC_admm]=FLAT_BIC(x,y,lon,lat,options)


% Input Options
if isfield(options,'intercept_type')==1
    if isempty(options.intercept_type)==0
        
    else
       options.intercept_type=1; % Use default value  
        
    end
else
    options.intercept_type=1; % Use default value  
end

if isfield(options,'lambda')==1
    if isempty(options.lambda)==0
        
    else
       options.lambda=10.^linspace(-6,1,300); % Use default value  
        
    end
else
    options.lambda=10.^linspace(-6,1,300); % Use default value
end

if isfield(options,'BIC')==1
    if isempty(options.BIC)==0
        
    else
       options.BIC=1; % Use default value  
        
    end
else
    options.BIC=1; % Use default value  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[n,p]=size(x);
Xg=zeros(n,n*p);


for j=1:p
    Xg(:,(j-1)*n+1:j*n)=diag(x(:,j));
end

% 计算初始beta
[aaa,BIC]=SCC(x,y,lon,lat,[]);
[~,index]=min(BIC);
beta_local=reshape(aaa(:,index),[n,p]);
beta_local = squeeze(beta_local);
if options.intercept_type==1
    beta_local = beta_local(:,1:end-1);
end

dc = pdist(beta_local);

% 缩放权重
min_ = 0.5; % 0.5
max_ = 2; % 2
max_beta = max(dc);
min_beta = min(dc);
new_dc = (dc-min_beta)/(max_beta-min_beta);
new_pi = min_+new_dc*(max_-min_);
pi = (squareform(new_pi)+1e-5).^(-1);


% 计算加权H
[~, H_hat]=FLAT_spanning_tree(pi,lon,lat,p,0.05);
G=Xg/H_hat;             % Xg为n x np, H_hat为np x np, 该式求解G \cdot H_hat = X_g

% 利用glmnet求解lasso
if options.intercept_type==0
    options.standardize = false;
    options.penalty_factor=ones(n*p,1);
    options.penalty_factor(end-p+1:end)=0;
    options.intr=false;
    FitInfo = glmnet(G,y,[],options);
    B=FitInfo.beta;
    beta=H_hat\B;
    [MSE]=SCC_fit_MSE(B,G,y,FitInfo.a0);
    k=FitInfo.df;
    if options.BIC==1
        % using BIC
        BIC=n*log(MSE)+k*log(n);
    else
        % using EBIC
        BIC_add=nan(length(k),1);
        for qqq=1:length(k)
            ccc1=n*p:-1:(n*p-k(qqq)+1);
            ccc2=1:k(qqq);
            BIC_add(qqq)=2*(sum(log(ccc1))-sum(log(ccc2)));
        end
        BIC=n*log(MSE)+k*log(n)+BIC_add;
    end
        
elseif options.intercept_type==1
    options.standardize = false;
    options.penalty_factor=ones(n*p-1,1);
    options.penalty_factor(end-p+2:end)=0;
    FitInfo = glmnet(G(:,1:end-1),y,[],options);
    B=FitInfo.beta;
    B(end+1,:)=0;
    FitInfo.beta=B;
    beta=H_hat\B;     % B为np x n_lambda，由不同lambda取值下的beta_hat的向量组成
    for t=1:length(B(1,:))
        beta(n*(p-1)+1:n*p,t)=beta(n*(p-1)+1:n*p,t)+FitInfo.a0(t);
    end
    [MSE]=SCC_fit_MSE(B,G,y,FitInfo.a0);
    k=FitInfo.df;
    if options.BIC==1
        % using BIC
        BIC=n*log(MSE)+k*log(n);
    else
        % using EBIC
        BIC_add=nan(length(k),1);
        for qqq=1:length(k)
            ccc1=n*p:-1:(n*p-k(qqq)+1);
            ccc2=1:k(qqq);
            BIC_add(qqq)=2*(sum(log(ccc1))-sum(log(ccc2)));
        end
        BIC=n*log(MSE)+k*log(n)+BIC_add;
    end
end
beta_admm = beta;
BIC_admm = BIC;

end
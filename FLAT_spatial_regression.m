function[beta_hat,MSE,BIAS,VAR]=FLAT_spatial_regression(x,y,lon,lat,beta,sim_num,options)

[n,p]=size(beta);
beta_hat=nan(sim_num,n,p);
%theta_hat=nan(sim_num,n,p);
%check=nan(sim_num,2);
for t=1:sim_num
    t
    % aaa is beta_hat.
    [aaa,BIC]=FLAT_BIC(squeeze(x(t,:,:)),squeeze(y(t,:))',lon,lat,options);
    %plot(x_lab,CRI)
    
    [~,index]=min(BIC);
    beta_hat(t,:,:)=reshape(aaa(:,index),[n,p]);
    
    % If you want to use this output, param 'fold_name' should be added
    % writematrix(squeeze(beta_hat(t,:,:)), strcat('./data/FLAT/',fold_name,'/',string(t),'.csv'));
end

MSE=0;
BIAS=0;
VAR=0;
if sim_num>1

    MSE=nan(sim_num,n,p);
    BIAS=nan(sim_num,n,p);
    VAR=nan(sim_num,n,p);

    for t=1:sim_num
        MSE(t,:,:)=(squeeze(beta_hat(t,:,:))-beta).^2;
        BIAS(t,:,:)=abs(squeeze(beta_hat(t,:,:))-beta);
        VAR(t,:,:)=(squeeze(beta_hat(t,:,:))-squeeze(mean(beta_hat))).^2;
    end

    MSE=squeeze(mean(MSE));
    BIAS=squeeze(mean(BIAS));
    VAR=squeeze(mean(VAR));
end
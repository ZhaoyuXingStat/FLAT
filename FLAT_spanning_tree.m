function[H_, H]=FLAT_spanning_tree(pi,lon,lat,p,dc)
n=length(lon);
lon=lon*ones(1,n);
lat=lat*ones(1,n);

tree_exist=0;
while tree_exist==0
    d=sqrt((lon-lon').^2+(lat-lat').^2);
    d(d>dc)=0;
    G = graph(d);
    T = minspantree(G);
    [index1_path,index2_path]=find(tril(T.adjacency));
    if isempty(index1_path)==1||length(index1_path)<n-1
        dc=dc*2;
    else
        tree_exist=1;
    end
end
    

H1=zeros(n-1,n); % (n-1)*n

ppp=sub2ind(size(H1),[1:n-1]',index1_path);
H1(ppp)=1;
ppp=sub2ind(size(H1),[1:n-1]',index2_path);
H1(ppp)=-1;
H_ = [H1;ones(1,n)./n];
% beta加权
index = [index1_path,index2_path];
weight = zeros(n-1,1);
for i=1:n-1
    weight(i,1)=pi(index(i,1),index(i,2));
end

H1 = H1.*weight;

H=zeros(n*p,n*p);
for j=1:p
    H((j-1)*n+1:j*n-1,(j-1)*n+1:j*n)=H1;
end
for j=1:p
   H(j*n,(j-1)*n+1:j*n)=0.1/n;
end
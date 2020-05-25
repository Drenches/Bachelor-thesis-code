function datapass(S,t,f,a)
data0=S(:,:,1);
data1=t(:,1);
data2=f(:,1);
for i=2:a
    data0=[data0;S(:,:,i)];
    data1=[data1;f(:,i)];
    data2=[data2;t(:,i)];
end
data=[data0,data1,data2];
xlswrite('data.xlsx',data,1);

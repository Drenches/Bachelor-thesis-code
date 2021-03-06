%% 确定基本参数
clear;
n=5;     %节点个数
m=50;    %分配循环次数
L=5;     %有效窗口长度
load('Sbn.mat', 'Sbn');

%% 设定范围内随机生成S和b序列
%Sbn=zeros(n,2,m);
% for i=1:n                   
%     ss=strat(m);
%     for j=1:m
%         if ss(j)~=1
%             Sbn(i,1,j)=0;
%             Sbn(i,2,j)=randperm(5,1);
%         else
%             Sbn(i,1,j)=randperm(5,1);
%             Sbn(i,2,j)=0;
%         end
%     end
% end
% for i=1:10
%     Sbn(n,:,i)=0;
% end
%% 单节点往期资源与需求计算
sums1=zeros(n,1,m);
sumb1=zeros(n,1,m);
for k=m-1:-1:1  
    for i=1:n   
        if Sbn(i,2,k)~=0
            if  k>L
            for j=k:-1:k-L  %窗口滑动
                sums1(i,1,k)=sums1(i,1,k)+Sbn(i,1,j);
                sumb1(i,1,k)=sumb1(i,1,k)+Sbn(i,2,j);
            end
            else
            for j=k:-1:1   %开始几块不滑动
                sums1(i,1,k)=sums1(i,1,k)+Sbn(i,1,j);
                sumb1(i,1,k)=sumb1(i,1,k)+Sbn(i,2,j);
            end
            end
        end
    end
end
%% 往期总贡献资源计算
for i=1:m
    sumt(i)=sum(sums1(:,:,i));
end
%% 当前可用总资源计算
sumb3=zeros(1,m);
sums3=zeros(1,m);
f1=zeros(n,m);
for i=m:-1:1
    sums3(i)=sum(Sbn(:,1,i));
    sumb3(i)=sum(Sbn(:,2,i));
    if sums3(i)>=sumb3(i)
        for j=1:n 
           f1(j,i)=Sbn(j,2,i);%涵盖的可以直接确定
        end
    end   
end
%% 初始分配
f2=zeros(n,m);
for i=m:-1:1
    for j=1:n
        if Sbn(j,2,i)~=0 && f1(j,i)==0
            f2(j,i)=(sums1(j,1,i)/sumt(i))*sums3(i)-0.1*sumb1(j,1,i);
        else
            f2(j,i)=0;
        end
    end
end
%% 二次分配
Real=zeros(n,2,m);
c=zeros(n,m);
sums2=zeros(1,m);
sums4=zeros(1,m);
sums5=zeros(1,m);
sumf=zeros(1,m);
sumf11=zeros(1,m);
ff=zeros(n,m);
for i=m:-1:1   %计算第一轮分配剩余的资源
    for j=1:n
        c(j,i)=f2(j,i)-Sbn(j,2,i);
        if c(j,i)>0 && f1(j,i)~=0
            sums2(i)=sums2(i)+c(j,i);
        end
        if f2(j,i)<0 
            sumf(i)=sumf(i)+(sums1(j,1,i)/sumt(i))*sums3(i);
            f2(j,i)=0;
        else
            sumf11(i)=sumf11(i)+0.1*sumb1(j,1,i);
        end
    end
end
sums5=sums2+sumf+sumf11;
for i=m:-1:1    %最终资源分配
    for j=1:n
        if f1(j,i)==0
           f2(j,i)=f2(j,i)+sums5(i)*(Sbn(j,2,i)/sumb3(i));
        end
    end
end
ff=f1+f2;
%% 输送数据结果
sat=zeros(n,m);
tot=zeros(1,n);
num=zeros(1,n);
for i=m:-1:1
    for j=1:n
        if Sbn(j,2,i)~=0
           sat(j,i)=ff(j,i)/Sbn(j,2,i);%各个时段满意度
        end
    end
end 
for j=1:n 
    tot(j)=sum(sat(j,:)~=0);
    num(j)=sum(sum(sat~=0));
    ave(j)=tot(j)/num(j);
end
%ave2(z/50)=var(ave);
% for j=1:n
%     for i=1:m
%         if sat(i)~=0
%             tot(j)=tot(j)+sat(i);
%             num=num+1
%         end
%     end
%     ave=tot(j)/
% end
% datapass(Sbn,sat,ff,m);
aa=ff(5,:);
for i=1:m
  bb(i)=Sbn(5,2,i);
  cc(i)=-Sbn(5,1,i);
  if  bb(i)<aa(i)
      aa(i)=bb(i);
  end
end
% plot(aa,'*-');
% hold;
% plot(bb,'o-');
% hk=[0.500000000000000,0.333333333333333,0.250000000000000,0.200000000000000,0.166666666666667,0.142857142857143,0.125000000000000,0.111111111111111,0.100000000000000,0.0909090909090909,0.0833333333333333,0.0769230769230769,0.0714285714285714,0.0666666666666667,0.0625000000000000,0.0588235294117647,0.0555555555555556,0.0526315789473684,0.0500000000000000,0.0476190476190476,0.0454545454545455,0.0434782608695652,0.0416666666666667,0.0400000000000000,0.0384615384615385,0.0370370370370370,0.0357142857142857,0.0344827586206897,0.0333333333333333,0.0322580645161290,0.0312500000000000,0.0303030303030303,0.0294117647058824,0.0285714285714286,0.0277777777777778,0.0270270270270270,0.0263157894736842,0.0256410256410256,0.0250000000000000,0.0243902439024390,0.0238095238095238,0.0232558139534884,0.0227272727272727,0.0222222222222222,0.0217391304347826,0.0212765957446809,0.0208333333333333,0.0204081632653061,0.0200000000000000,0.0196078431372549,0.0192307692307692,0.0188679245283019,0.0185185185185185,0.0181818181818182,0.0178571428571429,0.0175438596491228,0.0172413793103448,0.0169491525423729,0.0166666666666667,0.0163934426229508,0.0161290322580645,0.0158730158730159,0.0156250000000000,0.0153846153846154,0.0151515151515151,0.0149253731343284,0.0147058823529412,0.0144927536231884,0.0142857142857143,0.0140845070422535,0.0138888888888889,0.0136986301369863,0.0135135135135135,0.0133333333333333,0.0131578947368421,0.0129870129870130,0.0128205128205128,0.0126582278481013,0.0125000000000000,0.0123456790123457,0.0121951219512195,0.0120481927710843,0.0119047619047619,0.0117647058823530,0.0116279069767442,0.0114942528735632,0.0113636363636364,0.0112359550561798,0.0111111111111111,0.0109890109890110,0.0108695652173913,0.0107526881720430,0.0106382978723404,0.0105263157894737,0.0104166666666667,0.0103092783505155,0.0102040816326531,0.0101010101010101,0.0100000000000000];
%plot(hk,'*-');


        
            

  
        
    
    
        



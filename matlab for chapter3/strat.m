function p1=strat(col)
% col = 100;
lin = 1;
p1 = unifrnd(0,1,lin,col);%(0,1)均匀分布中随机抽取一些数
for i=1:col
	if p1(1,i)>=0.5 p1(1,i) = 1;
	else p1(1,i) = 0;
    end
end
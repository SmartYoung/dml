from __future__ import division
import numpy as np
import scipy as sp
pi=3.14159
def gauss(a,mu,sigma): # 假设连续特征变量遵循高斯分布
	#print sigma
	return 1/(np.sqrt(2*pi))*np.exp(-(a-mu)**2/(2*sigma**2))
class NBC:
	def __init__(self,X,y,Indicator=None):
		'''
		 X 			 a N*M matrix where M is the train case number
		 			 should be number both continous and descret feature
		 y			 class label for classification
		 Indicator 	 show whether the feature is continous(0) or descret(1)
		             continous in default

		'''
		self.X=np.array(X)
		#print self.X.dtype
		self.N,self.M=self.X.shape
		self.y=np.array(y).flatten(1)
		assert self.X.shape[1]==self.y.size

		self.labels=np.unique(y)  # 类别标签集合(实质为无重复元素的array)
		self.case={}  # 最终记录 case = {特征：{特征取值：{类别：条件概率}}}，采用嵌套字典存储条件概率table
		self.mu={}  # 记录各个连续特征变量的在不同类别下高斯分布参数 {特征：{类别：参数mu或sigma}
		self.sigma={}
		self.county={}  # 统计训练集中各个类别y=ys的计数 {类别：计数}
		if Indicator==None:  # 默认各特征为连续变量，否则根据实际indicator来标识
			self.Indicator=np.zeros((1,self.N)).flatten(1)
		else:
			self.Indicator=np.array(Indicator).flatten(1)
		assert self.Indicator.size==self.N
        
	def count(self,i,t,ys): # 对于所在类别ys的第i个特征，取值为t，进行计数！
		counts=0
		for p in range(self.M): # 对于第i个特征的所有取值遍历，当取值和类别满足给定条件时，增加计数
			if (self.X[i,p]==t) and (self.y[p]==ys): 
				counts+=1
		return counts
   
   # 训练时最多有4个for循环嵌套，复杂度O(特征数*离散特征取值数*类别数*样本数)
	def train(self):    # 计算每个特征的条件概率分布，各个特征分别处理，即一维随机变量的条件概率分布
		for ys in self.labels:  # 根据训练集数据类别y，对于每一个类别统计其相应计数，用于求类别的先验概率
			self.county.setdefault(ys)
			self.county[ys]=np.sum(self.y==ys) # 每一个类别的计数
		for i in range(self.N):  # 对于特征第i特征处理
			if self.Indicator[i]==1:  # 判断特征i是否为离散变量
				self.case.setdefault(i,{})
				self.case[i].setdefault('counts')  # 对于每一个特征i,注意数据的输入格式为N*M，N为特征数量
				self.case[i]['counts']=np.unique(self.X[i,:]).size  # 计算特征i的所有取值数量，用于计算条件概率
				for t in np.unique(self.X[i,:]):  # 对每一个特征所有可能取值，计算其所属类别的计数，用于计算条件概率
					for ys in self.labels:
						self.case[i].setdefault(t,{})
						self.case[i][t].setdefault(ys)
						self.case[i][t][ys]=self.count(i,t,ys)
			elif self.Indicator[i]==0:  # 判断特征i是否为连续变量
				self.mu.setdefault(i,{})  
				self.sigma.setdefault(i,{})
				for ys in self.labels:
					tempx=self.X[i,self.y==ys]  # 取出第i个特征，类别为ys的所有取值
					#print tempx
					self.sigma[i].setdefault(ys)
					self.sigma[i][ys]=np.std(tempx)  # 方差估计
					self.mu[i].setdefault(ys)
					self.mu[i][ys]=np.mean(tempx)  # 期望估计
					if self.sigma[i][ys]==0:  # 如果特征变量在类别ys下方差估计为0，则设其方差为1？不理解！
						self.sigma[i][ys]=1
	       
	def nb_predict(self,x,showdetail=False):  #对于一个输入特征向量x，进行类别预测
		x=x.flatten(1)
		maxp=0
		y=self.labels[0]
		for ys in self.labels:
			now=(self.county[ys]+1)/(self.M+self.labels.size)  #  类别先验概率，且进行了laplace平滑
			for i in range(self.N):  # 对于每一个特征变量
				if self.Indicator[i]==1:  # 对于离散特征变量，采用贝叶斯经验估计，见统计学习基础李航
					self.case[i].setdefault(x[i],{})  # case = {特征i:{ 取值x[i]：{} }}
					self.case[i][x[i]].setdefault(ys,0)  # case = {特征i:{ 取值x[i]：{ 类别ys：条件计数} }}
					now=now*((self.case[i][x[i]][ys]+1)/(self.county[ys]+self.case[i]['counts']))  # 根据计数计算条件概率*类别先验概率 = 后验概率，并进行了laplace平滑
				else:  # 对于连续特征变量，采用高斯分布计算条件概率，并乘以类别先验概率now计算后验概率
					now=now*gauss(x[i],self.mu[i][ys],self.sigma[i][ys])
			if now>maxp:  # 比较各类别ys下，后验概率大小，选择最大后验概率的类别返回
				maxp=now
				y=ys
			if showdetail:
				print now,ys
		return y
    
	def pred(self,Test_X,showdetail=False): # 预测一组输入特征向量，并预测
		Test_X=np.array(Test_X)
		test_y=[]
		for i in range(Test_X.shape[1]):
			test_y.append(self.nb_predict(Test_X[:,i],showdetail))
		return test_y

# 需改进的地方：
"""
1. 可读性
2. 预测步骤nb_predict中，离散变量条件概率的计算可以在训练时完成，这样可加快预测速度
3. 类的属性和方法设计，可以进一步封装优化，如nb_predict，count函数，可设计为helping函数
   连续变量的高斯分布假设可作为一个训练参数，可以假设为其他分布，同时相应估计方法需要修改

"""









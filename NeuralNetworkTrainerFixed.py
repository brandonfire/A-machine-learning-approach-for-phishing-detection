#!/usr/bin/python
# A complte Muti layer neural network for training and output the training model
# Author: Chengbin Hu
# Python: 2.7
 

from __future__ import absolute_import
from __future__ import division
import numpy as np





class MutiLayerNeuralNetwork:
	traininput = np.asarray([])
	trainoutput = np.asarray([])
	testinput = np.asarray([]) 
	testoutput = np.asarray([])
	weights = []
	bias = []
	def __init__(self, size,Tdata,activefunction="sigmoid"):
		self.traininput,self.trainoutput,self.testinput,self.testoutput = self.NNdata(Tdata)
		if(activefunction =="tahn"):
			self.fun = self.tahn
		elif(activefunction =="relu"):
			self.fun = self.relu
		else:
			self.fun = self.sigmoidf
		

		
		np.random.seed(2)
		#initialize the weights and bias of each layer of neural network with mean =(2*0.5 -1) = 0
		self.inputlen = size[0]
		self.outputlen = size[-1]
		self.layers = len(size)-1
		#self.sizes = size
		

		self.weights.append(2*np.random.random((self.inputlen,size[1])) - 1)
		for x in range(self.layers-1):
			self.weights.append(2*np.random.random((size[x+1],size[x+2])) - 1)
		

		for x in range(self.layers):
			self.bias.append(2*np.random.random() - 1)
		
	
	
	#first is the function of sigmoid. We have two parts, function for sigmoid forward output or for derivative of sigmoid back propagation
	def sigmoidf(self,x,deri=False):
		if(deri == True):
			return x*1.0*(1.0-x)
		return 1.0/(1+np.exp(-x))
	def tahn(self,x,deri=False):
		if(deri == True):
			return 1 - x * x
		else:
			return 2 / (1 + np.exp(-2 * x)) - 1
	def relu(self,x,deri=False):
		if(deri == True):
			 i = (x >= 0)
			 res= np.zeros(x.shape)
			 res[i] = 1.0
			 return res
		else:
			return np.max([x, np.zeros(x.shape)], axis=0)

	def softmax(self,x,z=None,der=False):
		if der == True:
			return z(z-x)
		else:
			t = np.sum(np.exp(x),1).repeat(x.shape[1])
			t.shape = x.shape
			return np.exp(x)/t

	def cross_entropy_loss(self, x, y):
	 	 return np.sum(-(y*np.log(self.softmax(x))))


	# This is predict data based on testdata. We must run this after data training process.
	def predictd(self):
		ldata = []
		ldata.append(self.testinput)
		for x in range(self.layers):
			#at test we use expectation 0.5 as mask
			if(x==1):
				#mask = np.random.randint(2, size=(len(self.weights[x]),1))
						#print("length:",len(self.weights[x]),"\n")
				dropweight = np.asarray(self.weights[x])*0.5
				ldata.append(self.fun(np.dot(ldata[x],dropweight)+self.bias[x]))
					#	pass
			else:
				ldata.append(self.fun(np.dot(ldata[x],self.weights[x])+self.bias[x]))
		ldata.append(self.softmax(ldata[-1]))
		l_error=abs(self.testoutput-ldata[-2])
		print "prediction:"
		print ldata[-2]
		print "Testoutput"
		print self.testoutput
		print "Error:"
		print l_error
		x = 0
		for i in l_error:
			if(abs(i)<0.5):
				x+=1
		print "accuracy"
		print x*1.0/len(l_error)
	
	
	#This is how to train the data. 
	def traind(self):
		#stochastic part
		totaldata = np.zeros((self.traininput.shape[0],self.traininput.shape[1]+1))
		for xx in range(self.traininput.shape[0]):
			for yy in range(self.traininput.shape[1]):
				totaldata[xx][yy] = self.traininput[xx][yy]
			totaldata[xx][-1] = self.trainoutput[xx][0]
		#shuffle the total data 
		np.random.shuffle(totaldata)
		blocksize = 20#we choose 20 as the blocksize
		eachBlock = [totaldata[j:j+blocksize]for j in range(0, len(totaldata), blocksize)]
		#print("total data is",eachBlock[0])
		#exit()
		for block in eachBlock:#stochastic gradient decent

			
			inputd = block[:,:-1]
			outputd = block[:,-1:]
			for iter in range(6000):
				ldata = []
				ldata.append(inputd)
				for x in range(self.layers):
					#dropout mask at first hidden layer
					if(x==1):
						mask = np.random.randint(2, size=(len(self.weights[x]),1))
						#print("length:",len(self.weights[x]),"\n")
						dropweight = np.asarray(self.weights[x])*mask
						ldata.append(self.fun(np.dot(ldata[x],dropweight)+self.bias[x]))
					#	pass
					else:
						ldata.append(self.fun(np.dot(ldata[x],self.weights[x])+self.bias[x]))
		
				#lastoutput = ldata[-1]
				#outsoftmax =  self.softmax(np.dot(ldata[x],self.weights[x])+self.bias[x])

				error = []
				delta = []

				er = self.cross_entropy_loss(ldata[-1],outputd)

				error.append(outputd-ldata[-1])
				if (iter% 1000) == 0:
						print "Error:",str(np.mean(np.abs(error[0])))
			

				delta.append(error[0] * self.fun(ldata[-1],True))
				for y in range(self.layers-1):
					error.append(delta[y].dot(self.weights[-(y+1)].T))
					delta.append(error[y+1] * self.fun(ldata[-(y+2)],True))
				for z in range(self.layers):
					self.weights[z] += np.dot(ldata[z].T,delta[-(z+1)])
					self.bias[z] += np.mean(delta[-(z+1)])
			
			
				
        

			
		print "Loss function:",er.sum()

	def output(self):#function for output neural network parameters.
		outb=np.asarray(self.bias)
		print outb.shape
		for x in range(self.layers):
			with file('./weights.out'+str(x), 'w') as outfile:
				np.savetxt(outfile,self.weights[x],delimiter=',')
		with file('./bias.out', 'w') as outfile:
			np.savetxt(outfile,outb,delimiter=',')
	
	
	
	def NNdata(self,Tdata):
		data = np.loadtxt(Tdata, dtype={'names': ('x1', 'x2', 'x3','x4', 'x5', 'x6', 'x7', 'x8', 'x9','x10','x11', 'x12', 'x13','x14', 'x15', 'x16', 'x17', 'x18', 'x19','x20','x21', 'x22', 'x23','x24', 'x25', 'x26', 'x27', 'x28', 'x29','x30','class'), 'formats': (np.float, np.float, np.float, np.float,np.float, np.float, np.float, np.float,np.float, np.float,np.float, np.float, np.float, np.float,np.float, np.float, np.float, np.float,np.float, np.float,np.float, np.float, np.float, np.float,np.float, np.float, np.float, np.float,np.float, np.float,np.float)}, delimiter=',')
		np.random.shuffle(data)
		output = []
		i = []
		ti = []
		toutput = []
		length = len(data)
		for x in range(length):
			if(x<length*0.9):
				i.append([data[x][0],data[x][1],data[x][2],data[x][3],data[x][4],data[x][5],data[x][6],data[x][7],data[x][8],data[x][9],data[x][10],data[x][11],data[x][12],data[x][13],data[x][14],data[x][15],data[x][16],data[x][17],data[x][18],data[x][19],data[x][20],data[x][21],data[x][22],data[x][23],data[x][24],data[x][25],data[x][26],data[x][27],data[x][28],data[x][29]])#because output is 1 or -1. we code to 1 or 0
				if(data[x][30] == 1):
					output.append(1)
				elif(data[x][30] == -1):
					output.append(0)
			else:
				ti.append([data[x][0],data[x][1],data[x][2],data[x][3],data[x][4],data[x][5],data[x][6],data[x][7],data[x][8],data[x][9],data[x][10],data[x][11],data[x][12],data[x][13],data[x][14],data[x][15],data[x][16],data[x][17],data[x][18],data[x][19],data[x][20],data[x][21],data[x][22],data[x][23],data[x][24],data[x][25],data[x][26],data[x][27],data[x][28],data[x][29]])#because output is 1 or -1. we code to 1 or 0
				if(data[x][30] == 1):
					toutput.append(1)
				elif(data[x][30] == -1):
					toutput.append(0)
		
		return np.asarray(i), np.asarray([output]).T, np.asarray(ti), np.asarray([toutput]).T

		
def testdata():
	
	size = [30,50,100,50,10,5,1]#This is the size of neural network. first item is input size, then neuons of each layer, the last one is output size
	N4 =  MutiLayerNeuralNetwork(size,"./PhishingM","sigmoid")# arguement size: the size of nural network,data: datapath,activefcuntion "sigmoid"
	N4.traind()
	N4.predictd()
	N4.output()

	
if __name__ == "__main__":
        testdata()	

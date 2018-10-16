import chainer
from chainer import Chain,Variable,cuda
import chainer.functions as F
import numpy as np
import cupy as xp
import Network
import matplotlib.pyplot as plt
from PIL import Image




count = 0

def check_accuracy(model, xs, ts):
    ys = model(xs)
    loss = F.mean_squared_error(ys, ts)
    ys = np.argmax(ys.data, axis=1)
    cors = (ys == ts)
    num_cors = sum(cors)
    accuracy = num_cors / ts.shape[0]
    return accuracy, loss

def massan(x,y):
	import __main__
	x = cuda.to_cpu(x.reshape(28,28))
	y = cuda.to_cpu(y.reshape(28,28))
	a = Image.fromarray(np.uint8(x*255))
	a.save("massans\\"+str(__main__.count)+"x.png")
	b = Image.fromarray(np.uint8(y*255))
	b.save("massans\\"+str(__main__.count)+"y.png")
	__main__.count+=1

def main():
	model = Network.MLP()

	gpu_device = 0
	cuda.get_device(gpu_device).use()
	model.to_gpu(gpu_device)

	oppai = chainer.optimizers.Adam()
	oppai.setup(model)

	train,test = chainer.datasets.get_mnist()
	train_x,train_t = train._datasets
	train_x = xp.array(train_x)
	train_t = train_x
	test_x,test_t = test._datasets
	test_x = xp.array(test_x)
	test_t = test_x

	bm = 100

	for i in range(10000):
		total_loss = 0
		for j in range(600):
			model.cleargrads()
			x = train_x[(j * bm):((j+1)*bm)]
			#t = train_t[(j * bm):((j+1)*bm)]
			t = Variable(xp.array(x,dtype =xp.float32 ))
			y = model(x)
			#print("t =",t.shape)
			#for k in range(len(t)):
			#	print("t_",k," =",type(t.data[k]))
			#print("y =",y.shape)
			loss = F.mean_squared_error(y,x)
			total_loss += loss.data*bm
			loss.backward()
			oppai.update()
		print("Epoch:%d loss:%f"%(i+1,total_loss/60000))
		for j in range(1):
			massan(x[j],y.data[j])
		#accuracy_train,loss_train = check_accuracy(model,train_x,train_t)
		#accuracy_test,_ = check_accuracy(model,test_x,test_t)




		#print("Epoch %d loss(train) = %f, accuracy(train) = %f, accuracy(test) = %f" % (i + 1, loss_train.data, accuracy_train, accuracy_test))

if __name__ == '__main__':
	main()
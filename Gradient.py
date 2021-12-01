import numpy as np
import matplotlib as matlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from scipy.sparse import spdiags
from scipy.sparse import kron
from mpl_toolkits.mplot3d import Axes3D
import time
import random
from sklearn.decomposition import PCA, NMF
import scipy.sparse as sp
import scipy.linalg as linalg
from numpy.random import default_rng
import matplotlib.patches as mpatches
import ot

var = 0.6;
##this is the covariance matrix
cov = [[var,  0], [0, var]];
##this is the precision matrix
pre = [[1/var, 0], [0, 1/var]];

##define the the mixture of gaussian function 
##it takes k: number of mixtures
##x -- 2d vector, the position in 2d plane
##vec mu k pairs of means
##vec w k-1 weight parameters
def model(x, k, mu, w):
	result = 0;
	for i in range(k-1):
		result = result + w[i]*multivariate_normal.pdf(x, mean=mu[i], cov=[[var,  0], [0, var]]);
	w_last = 1 - np.sum(w);
	result = result + w_last*multivariate_normal.pdf(x, mean=mu[k-1], cov=[[var, 0], [0, var]]);
	return result;

#it generates and returns random parameters:
#[k-1 weight paras, k means]
def generate_random_para(k):
	mu_x = 0; mu_y = 0;
	if (k<=90):
		mu_x = np.array(random.sample(range(5, 95), k))/10;
		mu_y = np.array(random.sample(range(5, 95), k))/10;
	else:
		mu_x = np.array(random.sample(range(50, 950), k))/100;
		mu_y = np.array(random.sample(range(50, 950), k))/100;
	means = [];
	for i in range(k):
		means.append(np.array([mu_x[i], mu_y[i]]));
	means = np.array(means);
	weights = [random.randrange(40, 60) for i in range(k)];
	weights = np.array(weights);
	weights = weights/np.sum(weights);
	weights = weights[0:k-1];
	return [means, weights];

#X is the mesh grid points we want to evaluate at
#mu, w is the current parameters
#ytruth is the truth values
def loss(X, k, mu, w, ytruth):
	ymodel = model(X, k, mu, w);
	error = 0.5*(ytruth-ymodel)**2;
	return np.sum(error);

##Q is the precision matrix
def expinner(x, mu, Q):
	inner = -0.5*np.einsum("ij, ij->i", (x-mu)@Q, x-mu);
	return np.exp(inner);  

##it returns the gradient of p w.r.t all the w_i
##returns a matrix of size (N-1)^2*(k-1)
def dpdw(x, mu, w, k):
	result = [];
	for i in range(k-1):
		result.append(multivariate_normal.pdf(x, mean=mu[i], cov=[[var, 0], [0, var]]));
	result = result - multivariate_normal.pdf(x, mean=mu[k-1], cov=[[var, 0], [0, var]]);
	result = np.array(result);
	result = np.transpose(result);
	return result;

##for two mixtures of gaussian
##argument w scalar, mu is mu1
def dpdmu11(x, mu, w):
	expt = expinner(x,  mu, pre);
	ret = (w/(2*np.pi*var**2))*(x[:, 0] - mu[0]);
	ret = ret * expt;
	return ret;

#the derivative dp/dmu1:
def dpdmu(w, x, mu):
	t1 = expinner(x, mu); t2 = (mu-x);
	return -w*(1/(2*np.pi))*np.multiply(t2, t1[:, np.newaxis]);

##w is scalar, mu is an array of two means
##returns the gradient direction of J
##returns [mu11, w] as the rho gradient
def gradient_w_mu11(x, w, mu, ytruth):
	mu11 = dpdmu11(x, mu[0], w);
	w2 = dpdw(x, mu, w, 2);
	mu11 = mu11.reshape(-1, 1);
	drhodtheta =  np.concatenate((mu11, w2), axis=1);
	dJdrho = model(x, 2, mu, np.array([w])) - ytruth;
	direction = dJdrho.T @ drhodtheta;
	return direction, drhodtheta;

##this function computes the l2 direction of w, mu11
def l2_dir_wmu11(x, w, mu, ytruth):
	grad, Y = gradient_w_mu11(x, w, mu, ytruth);
	q, r = linalg.qr(Y, mode='economic');
	drhoU = dUdrho(x, 2, mu, np.array([w]), ytruth);
	return np.linalg.solve(r, -q.T@drhoU);

##W2  natural direction of w and mu11
##w is scalar
def w2_dir_wmu11(x, w, mu, ytruth, A, N):
	stddir, drhodtheta = gradient_w_mu11(x, w, mu, ytruth);
	stddir = -stddir;
	B = Bmatrix(x, N, np.array([w]), mu, 2, A);
	B = B.toarray();
	q, r = linalg.qr(B.T, mode='economic');
	Z = np.linalg.solve(r.T, -drhodtheta);
	Y = np.matmul(q, Z);
	Gw2 = Y.T@Y;
	return np.linalg.inv(Gw2)@stddir;


#it returns a (N-1)^2 by 3k-1 matrix
def p_grad(X, mu, w, k, N):
	result = np.zeros(((N-1)**2, 3*k-1));
	result[:, 0:k-1] = dpdw(X, mu, w, k);
	for i in range(k-1):
		result[:, 2*i+k-1:2*i+k+1] = dpdmu(w[i], X, mu[i]);
	w_last = 1- np.sum(w);
	result[:, 3*k-3:3*k-1] = dpdmu(w_last, X, mu[k-1]);
	return result;

#it returns a (N-1)^2 by 2k matrix
#which is the gradient w.r.t mus only
def p_grad_mu(X, mu, w, k, N):
	result = np.zeros(((N-1)**2, 2*k));
	for i in range(k-1):
		result[:, 2*i:2*i+2] = dpdmu(w[i], X, mu[i]);
	w_last = 1- np.sum(w);
	result[:, 2*k-2:2*k] = dpdmu(w_last, X, mu[k-1]);
	return result;

#it returns the gradient of p together with the gradient of U
#returns [p_grad, u_grad]
def U_grad(X, mu, w, k, N, ytruth):
	p = model(X, k, mu, w) - ytruth;
	P = p_grad(X, mu, w, k, N);
	Q = p.T @ P;
	return [P, Q];

#discretization

#compute the matrix A that represents the process of taking divergence
#returns a 2*(N-1)**2 by (N-1)**2 matrix
def discretize(N, dx, dy):
	ones = np.ones(N-1);
	diags = np.array([-1, 1]);
	data = [-ones, ones];
	B = spdiags(data, diags, N-1, N-1).toarray();
	I = np.identity(N-1);
	A1 = kron(I, B).toarray()/(2*dx);
	A2 = kron(B, I).toarray()/(2*dy);
	return np.concatenate((A1, A2), axis=0);

##compute the matrix sigma
def sigma(X, N, w, mu, k):
	diagonal = 1/model(X, k, mu, w);
	data = np.array([diagonal]);
	diags = np.array([0]);
	sig = spdiags(data, diags, (N-1)**2, (N-1)**2).toarray();
	return sig;

##compute the matrix big sigma
##that is a concatenation of sigma
def Sigma(X, N, w, mu, k):
	sig = sigma(X, N, w, mu, k);
	I = np.identity(2);
	Sig = kron(I, sig).toarray();
	return Sig;

##compute the matrix C
def Cmatrix(X, N, w, mu, k):
	C = Sigma(X, N, w, mu, k);
	C = 2*C;
	C = np.linalg.inv(C);
	return C;

#it returns a bsr_matrix
def Bmatrix(X, N, w, mu, k, A):
	A = sp.bsr_matrix(A.T);
	diagonal = model(X, k, mu, w)**0.5;
	data = np.array([diagonal]);
	diags = np.array([0]);
	sig = spdiags(data, diags, (N-1)**2, (N-1)**2);
	I = sp.identity(2);
	Pinv = kron(I, sig);
	B = A@Pinv;
	return B;

def quick_sort(numseq, indexseq):
	size = len(numseq);
	if (size<=1):
		return [numseq, indexseq];
	pivot = size//2;
	pivotvalue = numseq.pop(pivot);
	midnum = [pivotvalue];
	midindex = [indexseq.pop(pivot)];
	leftnum = []; rightnum = [];
	leftindex = []; rightindex = [];
	for i in range(size-1):
		if (numseq[i] > pivotvalue):
			leftnum.append(numseq[i]);
			leftindex.append(indexseq[i]);
		else:
			rightnum.append(numseq[i]);
			rightindex.append(indexseq[i]);
	sortednuml, sortedindexl = quick_sort(leftnum, leftindex);
	sortednumr, sortedindexr = quick_sort(rightnum, rightindex);
	return [sortednuml+midnum+sortednumr, sortedindexl+midindex+sortedindexr];

#m is the order of apprximation, the less m is, the quicker the algo 
#but less accurate the return value is
#m must be <= 3k-1
def Natural_dir2(X, N, k, w, mu, ytruth, A, m):
	B = Bmatrix(X, N, w, mu, k, A);
	B = B.toarray();
	q, r= linalg.qr(B.T, mode = 'economic');
	gradp, gradu = U_grad(X, mu, w, k, N, ytruth);
	Z = np.linalg.solve(r.T, -gradp);
	Y = np.matmul(q, Z);
	Q, R = linalg.qr(Y, mode = 'economic')
	diag = abs(np.diagonal(R))
	numseq, indexseq = quick_sort(list(diag), list(range(len(diag))))
	Rtil = R[indexseq[0:m], :]
	Q1, R1 = linalg.qr(Rtil.T, mode = 'economic');
	z = np.linalg.solve(R1, Q1.T @ gradu);
	y = np.linalg.solve(R1.T, z);
	x = Q1 @ y;
	return x;

##define 2 functions that transfer parameters from 2 vectors to 1 single vector
def to1(mu, w):
	return np.concatenate((w, mu.flatten()))

def to2(para, k):
	w = para[0:k-1];
	mu = para[k-1:3*k-1];
	return [w, np.reshape(mu, (-1, 2))];

def Natural_GD_test(lr, initial, iteration, k, N, A, ytruth, savepath):
	para = initial;
	w_initial, mu_initial = to2(initial, k);
	trace = [para]; 
	tracevalue = [loss(X, k, mu_initial, w_initial, ytruth)];
	start = time.time();
	for i in range(iteration):
		print(i)
		w, mu = to2(para, k);
		direct = Natural_dir2(X, N, k, w, mu, ytruth, A, 5)
		para = para - lr*direct;
		tracevalue.append(loss(X, k, mu, w, ytruth));
		trace.append(para);
	end = time.time();
	trace = np.array(trace);
	weight_trace = trace[:, 0];
	mu1_trace = trace[:, 1:3];
	mu2_trace = trace[:, 3:5];
	#plot the trace of parameters
	fig, ax = plt.subplots(2, 2, figsize=(7, 7));
	ax[0][0].plot(weight_trace);
	ax[0][1].plot(mu1_trace[:, 0], mu1_trace[:, 1], label='mu1 trace');
	ax[0][1].plot(mu2_trace[:, 0], mu2_trace[:, 1], label = 'mu2 trace');
	ax[0][1].plot(1, 3, 'ro', label = 'mu1 truth', c = 'blue');
	ax[0][1].plot(3, 2, 'ro', label = 'mu2 truth', c = 'orange');
	ax[0][1].legend();
	ax[1][0].plot(tracevalue);
	ax[1][1].semilogy(tracevalue);
	ax[0][0].set_title("Trace of Weight");
	ax[0][1].set_title("Trace of mu1 and mu2 in 2d");
	ax[1][0].set_title("Loss value");
	ax[1][1].set_title("Semilogy of Loss value");
	ax[0][0].set_xlabel("Iteration");
	ax[0][0].set_ylabel("Weight Value");
	ax[0][1].set_xlabel("x direction");
	ax[0][1].set_ylabel("y direction");
	ax[1][0].set_xlabel("Iteration");
	ax[1][0].set_ylabel("Loss");
	ax[1][1].set_xlabel("Iteration");
	ax[1][1].set_ylabel("Loss");
	fig.tight_layout();
	print("Initial parameter values:");
	print("Weight: {}  mu1:[{}, {}]  mu2:[{}, {}]".format(initial[0], initial[1], initial[2], 
													 initial[3], initial[4]))
	print("Optimized parameter values:");
	print("Weight: {}  mu1:[{}, {}]  mu2:[{}, {}]".format(round(para[0], 2), round(para[1], 2), round(para[2], 2), 
													 round(para[3], 2), round(para[4], 2)));
	print("Time for {} steps is {} seconds".format(iteration, end-start));
	print("Average time for each step is {} seconds".format((end-start)/iteration));
	lo = tracevalue[-1];
	print("The loss is {}.".format(lo));
	plt.savefig(savepath, dpi=300);
	
def std_GD_test(lr, initial, iteration, k, N, A, ytruth, savepath):
	para = initial;
	w_initial, mu_initial = to2(initial, k);
	trace = [para];
	tracevalue = [loss(X, k, mu_initial, w_initial, ytruth)];
	start = time.time();
	for i in range(iteration):
		w, mu = to2(para, k);
		P, direct = U_grad(X, mu, w, k, N, ytruth);
		para = para - lr*direct;
		tracevalue.append(loss(X, k, mu, w, ytruth));
		trace.append(para);
	end = time.time();
	trace = np.array(trace);
	weight_trace = trace[:, 0];
	mu1_trace = trace[:, 1:3];
	mu2_trace = trace[:, 3:5];
	#plot the trace of parameters
	fig, ax = plt.subplots(2, 2, figsize=(7, 7));
	ax[0][0].plot(weight_trace);
	ax[0][1].plot(mu1_trace[:, 0], mu1_trace[:, 1], label='mu1 trace');
	ax[0][1].plot(mu2_trace[:, 0], mu2_trace[:, 1], label = 'mu2 trace');
	ax[0][1].plot(1, 3, 'ro', label = 'mu1 truth', c = 'blue');
	ax[0][1].plot(3, 2, 'ro', label = 'mu2 truth', c = 'orange');
	ax[0][1].legend();
	ax[1][0].plot(tracevalue);
	ax[1][1].semilogy(tracevalue);
	ax[0][0].set_title("Trace of Weight");
	ax[0][1].set_title("Trace of mu1 and mu2 in 2d");
	ax[1][0].set_title("Loss value");
	ax[1][1].set_title("Semilogy of Loss value");
	ax[0][0].set_xlabel("Iteration");
	ax[0][0].set_ylabel("Weight Value");
	ax[0][1].set_xlabel("x direction");
	ax[0][1].set_ylabel("y direction");
	ax[1][0].set_xlabel("Iteration");
	ax[1][0].set_ylabel("Loss");
	ax[1][1].set_xlabel("Iteration");
	ax[1][1].set_ylabel("Loss");
	fig.tight_layout();
	print("Initial parameter values:");
	print("Weight: {}  mu1:[{}, {}]  mu2:[{}, {}]".format(initial[0], initial[1], initial[2], 
													 initial[3], initial[4]))
	print("Optimized parameter values:");
	print("Weight: {}  mu1:[{}, {}]  mu2:[{}, {}]".format(round(para[0], 2), round(para[1], 2), round(para[2], 2), 
													 round(para[3], 2), round(para[4], 2)));
	print("Time for {} steps is {} seconds".format(iteration, end-start));
	print("Average time for each step is {} seconds".format((end-start)/iteration));
	lo = tracevalue[-1];
	print("The loss is {}.".format(lo));
	plt.savefig(savepath, dpi=300)


def dUdrho(x, k, mu, w, ytruth):
	result = model(x, k, mu, w) - ytruth;
	return result;

##calculate the l2 natural gd direction without any reduction
def l2_Natural_dirFull(X, mu, w, k, N, ytruth):
	J= p_grad(X, mu, w, k, N)
	q, r= linalg.qr(J, mode = 'economic');
	drhoU = dUdrho(X, k, mu, w, ytruth);
	return np.linalg.solve(r, -q.T @ drhoU);

def l2_Natural_GD_test(lr, initial, iteration, k, N, A, ytruth, savepath):
	para = initial;
	w_initial, mu_initial = to2(initial, k);
	trace = [para]; 
	tracevalue = [loss(X, k, mu_initial, w_initial, ytruth)];
	start = time.time();
	for i in range(iteration):
		w, mu = to2(para, k);
		direct = l2_Natural_dirFull(X, mu, w, k, N, ytruth);
		para = para + lr*direct;
		tracevalue.append(loss(X, k, mu, w, ytruth));
		trace.append(para);
	end = time.time();
	trace = np.array(trace);
	weight_trace = trace[:, 0];
	mu1_trace = trace[:, 1:3];
	mu2_trace = trace[:, 3:5];
	#plot the trace of parameters
	fig, ax = plt.subplots(2, 2, figsize=(7, 7));
	ax[0][0].plot(weight_trace);
	ax[0][1].plot(mu1_trace[:, 0], mu1_trace[:, 1], label='mu1 trace');
	ax[0][1].plot(mu2_trace[:, 0], mu2_trace[:, 1], label = 'mu2 trace');
	ax[0][1].plot(1, 3, 'ro', label = 'mu1 truth', c = 'blue');
	ax[0][1].plot(3, 2, 'ro', label = 'mu2 truth', c = 'orange');
	ax[0][1].legend();
	ax[1][0].plot(tracevalue);
	ax[1][1].semilogy(tracevalue);
	ax[0][0].set_title("Trace of Weight");
	ax[0][1].set_title("Trace of mu1 and mu2 in 2d");
	ax[1][0].set_title("Loss value");
	ax[1][1].set_title("Semilogy of Loss value");
	ax[0][0].set_xlabel("Iteration");
	ax[0][0].set_ylabel("Weight Value");
	ax[0][1].set_xlabel("x direction");
	ax[0][1].set_ylabel("y direction");
	ax[1][0].set_xlabel("Iteration");
	ax[1][0].set_ylabel("Loss");
	ax[1][1].set_xlabel("Iteration");
	ax[1][1].set_ylabel("Loss");
	fig.tight_layout();
	print("Initial parameter values:");
	print("Weight: {}  mu1:[{}, {}]  mu2:[{}, {}]".format(initial[0], initial[1], initial[2], 
													 initial[3], initial[4]))
	print("Optimized parameter values:");
	print("Weight: {}  mu1:[{}, {}]  mu2:[{}, {}]".format(round(para[0], 2), round(para[1], 2), round(para[2], 2), 
													 round(para[3], 2), round(para[4], 2)));
	print("Time for {} steps is {} seconds".format(iteration, end-start));
	print("Average time for each step is {} seconds".format((end-start)/iteration));
	lo = tracevalue[-1];
	print("The loss is {}.".format(lo));
	plt.savefig(savepath, dpi=300);

#the derivative dp/dmu1:
def dpdmu1(w, x, mu1):
	t1 = expinner(x, mu1); t2 = (mu1-x);
	return -w*(1/(2*np.pi))*np.multiply(t2, t1[:, np.newaxis]);
	
#the derivative dp/dmu2:
def dpdmu2(w, x, mu2):
	t1 = expinner(x, mu2); t2 = (mu2-x);
	return (w-1)*(1/(2*np.pi))*np.multiply(t2, t1[:, np.newaxis]);

##this function returns the gradient of p and U w.r.t mu1 only
def gradient_mu1_all(X, w, mu1, mu2, ytruth):
	p = model(X, 2, np.array([mu1, mu2]), np.array([w])) - ytruth;
	dpmu1 = dpdmu1(w, X, mu1);
	Q = np.sum(np.multiply(dpmu1, p[:, np.newaxis]), axis=0);
	return [dpmu1, Q];

##this function returns the gradient of p and U w.r.t mu2 only
def gradient_mu2_all(X, w, mu1, mu2, ytruth):
	p = model(X, 2, np.array([mu1, mu2]), np.array([w])) - ytruth;
	dpmu2 = dpdmu2(w, X, mu2);
	Q = np.sum(np.multiply(dpmu2, p[:, np.newaxis]), axis=0);
	return [dpmu2, Q];

#m is the order of apprximation, the less m is, the quicker the algo 
#but less accurate the return value is
#m must be <= 3k-1
def Natural_dir_mu1(X, N, w, mu, ytruth, A, m):
	B = Bmatrix(X, N, w, mu, 2, A);
	B = B.toarray();
	q, r= linalg.qr(B.T, mode = 'economic');
	gradp, gradu = gradient_mu1_all(X, w[0], mu[0], mu[1], ytruth);
	Z = np.linalg.solve(r.T, -gradp);
	Y = np.matmul(q, Z);
	Q, R = linalg.qr(Y, mode = 'economic')
	diag = abs(np.diagonal(R))
	numseq, indexseq = quick_sort(list(diag), list(range(len(diag))))
	Rtil = R[indexseq[0:m], :]
	Q1, R1 = linalg.qr(Rtil.T, mode = 'economic');
	z = np.linalg.solve(R1, Q1.T @ gradu);
	y = np.linalg.solve(R1.T, z);
	x = Q1 @ y;
	return x;

#m is the order of apprximation, the less m is, the quicker the algo 
#but less accurate the return value is
#m must be <= 3k-1
def Natural_dir_mu2(X, N, w, mu, ytruth, A, m):
	B = Bmatrix(X, N, w, mu, 2, A);
	B = B.toarray();
	q, r= linalg.qr(B.T, mode = 'economic');
	gradp, gradu = gradient_mu2_all(X, w[0], mu[0], mu[1], ytruth);
	Z = np.linalg.solve(r.T, -gradp);
	Y = np.matmul(q, Z);
	Q, R = linalg.qr(Y, mode = 'economic')
	diag = abs(np.diagonal(R))
	numseq, indexseq = quick_sort(list(diag), list(range(len(diag))))
	Rtil = R[indexseq[0:m], :]
	Q1, R1 = linalg.qr(Rtil.T, mode = 'economic');
	z = np.linalg.solve(R1, Q1.T @ gradu);
	y = np.linalg.solve(R1.T, z);
	x = Q1 @ y;
	return x;

##calculate the l2 natural gd direction without any reduction
##w is an array of weights, mu is an array of means
def l2_Natural_dirFull_mu1(X, mu, w, N, ytruth):
	J = dpdmu1(w[0], X, mu[0]);
	q, r= linalg.qr(J, mode = 'economic');
	drhoU = dUdrho(X, 2, mu, w, ytruth);
	return np.linalg.solve(r, -q.T @ drhoU);

##calculate the l2 natural gd direction without any reduction
##w is an array of weights, mu is an array of means
def l2_Natural_dirFull_mu2(X, mu, w, N, ytruth):
	J = dpdmu2(w[0], X, mu[1]);
	q, r= linalg.qr(J, mode = 'economic');
	drhoU = dUdrho(X, 2, mu, w, ytruth);
	return np.linalg.solve(r, -q.T @ drhoU);

##constant grid
##this is the grid that is used to plot the loss function 
grid = 0.1;

##it plots the loss function with fixed w, mu2
##it also plots the convergence history of mu1 given the trace and tracevalue paras
def plot_loss_mu1(X, w, mu2, grid, xrange, yrange, trace, tracevalue, ytruth, savepath1, savepath2):
	x = np.arange(xrange[0],xrange[1],grid);
	y = np.arange(yrange[0],yrange[1],grid);
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
		for j in range(y.size):
			Z[i, j] = loss(X, 2, np.array([np.array([grid*i+xrange[0], grid*j+yrange[0]]), mu2]), np.array([w]), ytruth);
	fig = plt.figure(figsize=(12, 8));
	axes = fig.gca(projection ='3d');
	axes.view_init(elev=50, azim=10);
	axes.plot(trace[:,1], trace[:,0], tracevalue, '*-', c = 'red')
	axes.plot_surface(P, Q, Z, alpha = 0.8);
	print(trace.shape);
	print(tracevalue.shape);
	plt.show();
	fig, ax = plt.subplots(figsize=(7, 7));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/40);
	CS = ax.contour(Q,P,Z,levels = level);
	ax.clabel(CS, inline=True, fontsize=10);
	ax.plot(trace[:, 0], trace[:, 1], "*-", c='red');
	ax.set_title('Loss Function w.r.t mu1')
	plt.savefig(savepath1, dpi=300);
	fig, ax = plt.subplots(1, 2, figsize=(8, 4));
	ax[0].plot(tracevalue, label = 'the loss value');
	ax[1].semilogy(tracevalue, label = 'semilogy loss value')
	ax[0].legend();
	ax[1].legend();
	ax[0].set_xlabel("number of iteration")
	ax[1].set_xlabel("number of iteration")
	ax[0].set_ylabel("Loss value")
	fig.suptitle('Loss along the Trace');
	plt.savefig(savepath2, dpi=300);
	
def plot_loss_mu2(X, w, mu1, grid, xrange, yrange, trace, tracevalue, ytruth, savepath1, savepath2):
	x = np.arange(xrange[0],xrange[1],grid);
	y = np.arange(yrange[0],yrange[1],grid);
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
		for j in range(y.size):
			Z[i, j] = loss(X, 2, np.array([mu1, np.array([grid*i+xrange[0], grid*j+yrange[0]])]), 
						  np.array([w]), ytruth);
	fig = plt.figure(figsize=(12, 8));
	axes = fig.gca(projection ='3d');
	axes.view_init(elev=50, azim=10);
	axes.scatter(trace[:,1], trace[:,0], tracevalue, '*-' ,c = 'red')
	axes.plot_surface(P, Q, Z, alpha = 0.8);
	print(trace.shape);
	print(tracevalue.shape);
	plt.show();
	fig, ax = plt.subplots(figsize=(7, 7));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/40);
	CS = ax.contour(Q,P,Z,levels = level);
	ax.clabel(CS, inline=True, fontsize=10);
	ax.plot(trace[:, 0], trace[:, 1], "*-", c='red');
	ax.set_title('Loss Function w.r.t mu2')
	plt.savefig(savepath1, dpi=300);
	fig, ax = plt.subplots(1, 2, figsize=(8, 4));
	ax[0].plot(tracevalue, label = 'the loss value');
	ax[1].semilogy(tracevalue, label = 'semilogy loss value')
	ax[0].legend();
	ax[1].legend();
	ax[0].set_xlabel("number of iteration")
	ax[1].set_xlabel("number of iteration")
	ax[0].set_ylabel("Loss value")
	fig.suptitle('Loss along the Trace');
	plt.savefig(savepath2, dpi=300);

#w and mu11 are not fixed, others are fixed
def plot_wmu11(X, mu2_fix, mu12, ytruth, xrange, yrange, N):
	dx = (xrange[1]-xrange[0])/N;
	dy = (yrange[1]-yrange[0])/N;
	x = np.arange(xrange[0],xrange[1],dx);
	y = np.arange(yrange[0],yrange[1],dy);
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
		for j in range(y.size):
			w = yrange[0]+dy*j;
			mu11 = xrange[0]+dx*i;
			Z[i, j] = loss(X, 2, np.array([np.array([mu11, mu12]), mu2_fix]), 
									np.array([w]), ytruth);
	Z=Z.T;
	fig, ax = plt.subplots(figsize=(14, 7));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/40);
	CS = ax.contour(P,Q,Z,levels = level);
	ax.clabel(CS, inline=True, fontsize=10);
	

##natural gradient descent on mu1 only with fixed w and mu2
def Natural_GD_mu1(X, N, A, w, mu2, lr, initial, iteration, plotrange, ytruth, savepath1, savepath2):
	para = initial; trace = [para]; tracevalue = [loss(X, 2, np.array([para, mu2]), np.array([w]), ytruth)];
	start = time.time();
	for i in range(iteration):
		print(i)
		direct = Natural_dir_mu1(X, N, np.array([w]), np.array([para, mu2]), ytruth, A, 2);
		para = para - lr*direct;
		trace.append(para);
		tracevalue.append(loss(X, 2, np.array([para, mu2]), np.array([w]), ytruth));
	end = time.time();
	trace = np.array(trace); tracevalue = np.array(tracevalue);
	plot_loss_mu1(X, w, mu2, grid, plotrange[0], plotrange[1],trace, tracevalue, ytruth, savepath1, savepath2);
	print("Initial parameter values:");
	print("mu1:[{},{}] ".format(initial[0], initial[1]))
	print("Optimized parameter values:");
	print("mu1:[{}, {}] ".format(round(para[0], 2), round(para[1], 2)));
	print("Time for {} steps is {} seconds.".format(iteration, end-start));
	print("The average time for each step is {} seconds.".format((end-start)/iteration));
	return [trace, tracevalue];
	
##natural gradient descent on mu2 only with fixed w and mu1
def Natural_GD_mu2(X, N, A, w, mu1, lr, initial, iteration, plotrange, ytruth, savepath1, savepath2):
	para = initial; trace = [para]; tracevalue = [loss(X, 2, np.array([mu1, para]), np.array([w]), ytruth)];
	start = time.time();
	for i in range(iteration):
		print(i)
		direct = Natural_dir_mu2(X, N, np.array([w]), np.array([mu1, para]), ytruth, A, 2);
		para = para - lr*direct;
		trace.append(para);
		tracevalue.append(loss(X, 2, np.array([mu1, para]), np.array([w]), ytruth));
	end = time.time();
	trace = np.array(trace); tracevalue = np.array(tracevalue);
	plot_loss_mu2(X, w, mu1, grid, plotrange[0], plotrange[1], trace, tracevalue, ytruth, savepath1, savepath2);
	print("Initial parameter values:");
	print("mu2:[{},{}] ".format(initial[0], initial[1]))
	print("Optimized parameter values:");
	print("mu2:[{}, {}] ".format(round(para[0], 2), round(para[1], 2)));
	print("Time for {} steps is {} seconds.".format(iteration, end-start));
	print("The average time for each step is {} seconds.".format((end-start)/iteration));
	return [trace, tracevalue];
	
##std gradient descent on mu1 only with fixed w and mu2
def std_GD_mu1(X, w, mu2, lr, initial, iteration, plotrange, ytruth, savepath1, savepath2):
	para = initial; trace = [para]; tracevalue = [loss(X, 2, np.array([para, mu2]), np.array([w]), ytruth)];
	start = time.time();
	for i in range(iteration):
		p, direct = gradient_mu1_all(X, w, para, mu2, ytruth)
		para = para - lr*direct;
		trace.append(para);
		tracevalue.append(loss(X, 2, np.array([para, mu2]), np.array([w]), ytruth));
	end = time.time();
	trace = np.array(trace); tracevalue = np.array(tracevalue);
	plot_loss_mu1(X, w, mu2, grid, plotrange[0], plotrange[1],trace, tracevalue, ytruth, savepath1, savepath2);
	print("Initial parameter values:");
	print("mu1:[{},{}] ".format(initial[0], initial[1]))
	print("Optimized parameter values:");
	print("mu1:[{}, {}] ".format(round(para[0], 2), round(para[1], 2)));
	print("Time for {} steps is {} seconds.".format(iteration, end-start));
	print("The average time for each step is {} seconds.".format((end-start)/iteration));
	return [trace, tracevalue];
	
##std gradient descent on mu1 only with fixed w and mu2
def std_GD_mu2(X, w, mu1, lr, initial, iteration, plotrange, ytruth, savepath1, savepath2):
	para = initial; trace = [para]; tracevalue = [loss(X, 2, np.array([mu1, para]), np.array([w]), ytruth)];
	start = time.time();
	for i in range(iteration):
		p, direct = gradient_mu2_all(X, w, mu1, para, ytruth)
		para = para - lr*direct;
		trace.append(para);
		tracevalue.append(loss(X, 2, np.array([mu1, para]), np.array([w]), ytruth));
	end = time.time();
	trace = np.array(trace); tracevalue = np.array(tracevalue);
	plot_loss_mu2(X, w, mu1, grid, plotrange[0], plotrange[1],trace, tracevalue, ytruth, savepath1, savepath2);
	print("Initial parameter values:");
	print("mu1:[{},{}] ".format(initial[0], initial[1]))
	print("Optimized parameter values:");
	print("mu1:[{}, {}] ".format(round(para[0], 2), round(para[1], 2)));
	print("Time for {} steps is {} seconds.".format(iteration, end-start));
	print("The average time for each step is {} seconds.".format((end-start)/iteration));
	return [trace, tracevalue];
	
##l2 natural gradient descent on mu1 only with fixed w and mu2
def l2_Natural_GD_mu1(X, N, w, mu2, lr, initial, iteration, plotrange, ytruth, savepath1, savepath2):
	para = initial; trace = [para]; tracevalue = [loss(X, 2, np.array([para, mu2]), np.array([w]), ytruth)];
	start = time.time();
	for i in range(iteration):
		direct = l2_Natural_dirFull_mu1(X, np.array([para, mu2]), np.array([w]), N, ytruth);
		para = para + lr*direct;
		trace.append(para);
		tracevalue.append(loss(X, 2, np.array([para, mu2]), np.array([w]), ytruth));
	end = time.time();
	trace = np.array(trace); tracevalue = np.array(tracevalue);
	plot_loss_mu1(X, w, mu2, grid, plotrange[0], plotrange[1],trace, tracevalue, ytruth, savepath1, savepath2);
	print("Initial parameter values:");
	print("mu1:[{},{}] ".format(initial[0], initial[1]))
	print("Optimized parameter values:");
	print("mu1:[{}, {}] ".format(round(para[0], 2), round(para[1], 2)));
	print("Time for {} steps is {} seconds.".format(iteration, end-start));
	print("The average time for each step is {} seconds.".format((end-start)/iteration));
	return [trace, tracevalue];

##l2 natural gradient descent on mu1 only with fixed w and mu1
def l2_Natural_GD_mu2(X, N, w, mu1, lr, initial, iteration, plotrange, ytruth, savepath1, savepath2):
	para = initial; trace = [para]; tracevalue = [loss(X, 2, np.array([mu1, para]), np.array([w]), ytruth)];
	start = time.time();
	for i in range(iteration):
		direct = l2_Natural_dirFull_mu2(X, np.array([mu1, para]), np.array([w]), N, ytruth);
		para = para + lr*direct;
		trace.append(para);
		tracevalue.append(loss(X, 2, np.array([mu1, para]), np.array([w]), ytruth));
	end = time.time();
	trace = np.array(trace); tracevalue = np.array(tracevalue);
	plot_loss_mu2(X, w, mu1, grid, plotrange[0], plotrange[1],trace, tracevalue, ytruth, savepath1, savepath2);
	print("Initial parameter values:");
	print("mu2:[{},{}] ".format(initial[0], initial[1]))
	print("Optimized parameter values:");
	print("mu2:[{}, {}] ".format(round(para[0], 2), round(para[1], 2)));
	print("Time for {} steps is {} seconds.".format(iteration, end-start));
	print("The average time for each step is {} seconds.".format((end-start)/iteration));
	return [trace, tracevalue];


##this function compute three distances of the trace
##argument trace is the array of mu1 
##argument fix_mu2 is the fixed mu2
##argument fix_w is the fixed weight 
def compute_length_mu1(X, N, A, trace, fix_mu2, fix_w):
	stdlength = 0; w2length = 0; l2length = 0;
	for i in range(len(trace)-1):
		print(i);
		theta_diff = trace[i+1] - trace[i];
		B = Bmatrix(X, 31, np.array(fix_w), np.array([trace[i+1], fix_mu2]), 2, A);
		B = B.toarray();
		q, r= linalg.qr(B.T, mode = 'economic');
		gradp = dpdmu1(np.array(fix_w), X, trace[i+1]);
		Z = np.linalg.solve(r.T, -gradp);
		Y = np.matmul(q, Z);
		stdlength+=np.linalg.norm(theta_diff, 2);
		w2length+=np.linalg.norm(Y@theta_diff, 2);
		l2length+=np.linalg.norm(gradp@theta_diff, 2);
	return [stdlength, w2length, l2length];

def compute_length_mu2(X, N, A, trace, fix_mu1, fix_w):
	stdlength = 0; w2length = 0; l2length = 0;
	for i in range(len(trace)-1):
		print(i);
		theta_diff = trace[i+1] - trace[i];
		B = Bmatrix(X, 31, np.array(fix_w), np.array([fix_mu1, trace[i+1]]), 2, A);
		B = B.toarray();
		q, r= linalg.qr(B.T, mode = 'economic');
		gradp = dpdmu2(np.array(fix_w), X, trace[i+1]);
		Z = np.linalg.solve(r.T, -gradp);
		Y = np.matmul(q, Z);
		stdlength+=np.linalg.norm(theta_diff, 2);
		w2length+=np.linalg.norm(Y@theta_diff, 2);
		l2length+=np.linalg.norm(gradp@theta_diff, 2);
	return [stdlength, w2length, l2length];

## xrange and yrange must be a 11 by 11 square
## w is a number, the weight
def Vector_field_stdmu1(X, mu2_fix, w, ytruth, trace='', savpath='', xrange = [-3.5, 8.5], yrange = [-3.5, 8.5]):
	matlib.rcParams['text.usetex'] = True;
	x = np.arange(xrange[0],xrange[1],grid);
	y = np.arange(yrange[0],yrange[1],grid);
	plt.rcParams.update({'font.size': 20})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
			for j in range(y.size):
				Z[i, j] = loss(X, 2, np.array([np.array([grid*i+xrange[0], grid*j+yrange[0]]), mu2_fix]), 
							   np.array([w]), ytruth);
	fig, ax = plt.subplots(figsize=(10, 10));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/40);
	CS = ax.contour(Q,P,Z,levels = level, zorder = 0);
	ax.clabel(CS, inline=True, fontsize=10);
	for i in np.arange(xrange[0], xrange[1], 0.5):
		print(i);
		for j in np.arange(yrange[0], yrange[1], 0.5):
			mu1 = np.array([i, j]);
			p, direct = gradient_mu1_all(X, np.array([w]), mu1, mu2_fix, ytruth);
			length = np.linalg.norm(direct);
			direct = -direct/length;
			plt.arrow(i, j, direct[0]/3, direct[1]/3, width=0.01, fc='blue', ec='blue', head_length = 0.1, head_width = 0.05);
	if (trace != ''):
		plt.plot(trace[:, 0], trace[:, 1], '*-', c='r', label = 'std GD trace', zorder = 5);
		plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
		plt.text(trace[0][0]-2, trace[0][1]+0.3, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_1$: $x$ direction");
	plt.ylabel(r"$\mu_1$: $y$ direction");
	plt.legend();
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300);

##plot the std vector field of w and mu11
##Nx the number of arrows in x direction, Ny the number of arrows in y direction
def VF_stdwmu11(X, mu2_fix, mu12, ytruth, trace='', savpath='', xrange = [-3, 4], yrange = [0, 1], Nx=35, Ny=5):
	matlib.rcParams['text.usetex'] = True;
	dx = (xrange[1]-xrange[0])/45;
	dy = (yrange[1]-yrange[0])/45;
	x = np.arange(xrange[0],xrange[1],dx);
	y = np.arange(yrange[0],yrange[1],dy);
	plt.rcParams.update({'font.size': 30})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
		for j in range(y.size):
			w = yrange[0]+dy*j;
			mu11 = xrange[0]+dx*i;
			Z[i, j] = loss(X, 2, np.array([np.array([mu11, mu12]), mu2_fix]), np.array([w]), ytruth);
	Z=Z.T;
	fig, ax = plt.subplots(figsize=(35, 5));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/20);
	CS = ax.contour(P,Q,Z,levels = level);
	ax.clabel(CS, inline=True, fontsize=10);
	gridx = (xrange[1]-xrange[0])/Nx; gridy = (yrange[1]-yrange[0])/Ny;
	for i in np.arange(xrange[0], xrange[1], gridx):
		for j in np.arange(yrange[0], yrange[1]+ 0.01, gridy):
			w = j; 
			mu11 = i;
			direct, p = gradient_w_mu11(X, w, np.array([[mu11, mu12], mu2_fix]), ytruth);
			length = np.linalg.norm(direct);
			direct = -direct/length;
			plt.arrow(i, j, direct[0]/20, direct[1]/20, width=0.005, fc='blue', ec='blue', head_length = 0.03, head_width = 0.02);
	if (trace != ''):
		plt.plot(trace[:, 0], trace[:, 1], '*-', c='r', label = 'std GD trace', zorder = 5);
		plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
		plt.text(trace[0][0]-0.5, trace[0][1]+0.1, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_{11}$: $x$ direction");
	plt.ylabel(r"$w$: $y$ direction");
	plt.legend(loc='upper left', bbox_to_anchor=(0, 1.5));
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300, bbox_inches = "tight");

##plot the l2 vector field of w, mu11
def VF_l2wmu11(X, mu2_fix, mu12, ytruth, trace='', savpath='', xrange = [-3, 4], yrange = [0, 1], Nx=35, Ny=5):
	matlib.rcParams['text.usetex'] = True;
	dx = (xrange[1]-xrange[0])/45;
	dy = (yrange[1]-yrange[0])/45;
	x = np.arange(xrange[0],xrange[1],dx);
	y = np.arange(yrange[0],yrange[1],dy);
	plt.rcParams.update({'font.size': 30})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
		for j in range(y.size):
			w = yrange[0]+dy*j;
			mu11 = xrange[0]+dx*i;
			Z[i, j] = loss(X, 2, np.array([np.array([mu11, mu12]), mu2_fix]), np.array([w]), ytruth);
	Z=Z.T;
	fig, ax = plt.subplots(figsize=(35, 5));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/20);
	CS = ax.contour(P,Q,Z,levels = level);
	ax.clabel(CS, inline=True, fontsize=10);
	gridx = (xrange[1]-xrange[0])/Nx; gridy = (yrange[1]-yrange[0])/Ny;
	for i in np.arange(xrange[0], xrange[1], gridx):
		for j in np.arange(yrange[0]+0.01, yrange[1]+ 0.01, gridy):
			w = j; 
			mu11 = i;
			direct = l2_dir_wmu11(X, w, np.array([[mu11, mu12], mu2_fix]), ytruth);
			length = np.linalg.norm(direct);
			direct = direct/length;
			plt.arrow(i, j, direct[0]/20, direct[1]/20, width=0.005, fc='blue', ec='blue', head_length = 0.03, head_width = 0.02);
	if (trace != ''):
		plt.plot(trace[:, 0], trace[:, 1], '*-', c='r', label = r'$L^2$ natural GD trace', zorder = 5);
		plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
		plt.text(trace[0][0]-.5, trace[0][1]+.1, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_{11}$: $x$ direction");
	plt.ylabel(r"$w$: $y$ direction");
	plt.legend(loc='upper left', bbox_to_anchor=(0, 1.5));
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300, bbox_inches = "tight");


##plot the w2 vector field of w, mu11
##savpath2 is for saving the arrow directions
def VF_w2wmu11(X, N, A, mu2_fix, mu12, ytruth, trace='', savpath='', savpath2='', xrange = [-3, 4], yrange = [0, 1], Nx=35, Ny=5):
	matlib.rcParams['text.usetex'] = True;
	dx = (xrange[1]-xrange[0])/45;
	dy = (yrange[1]-yrange[0])/45;
	x = np.arange(xrange[0],xrange[1],dx);
	y = np.arange(yrange[0],yrange[1],dy);
	plt.rcParams.update({'font.size': 30})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
		for j in range(y.size):
			w = yrange[0]+dy*j;
			mu11 = xrange[0]+dx*i;
			Z[i, j] = loss(X, 2, np.array([np.array([mu11, mu12]), mu2_fix]), np.array([w]), ytruth);
	Z=Z.T;
	fig, ax = plt.subplots(figsize=(35, 5));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/20);
	CS = ax.contour(P,Q,Z,levels = level);
	ax.clabel(CS, inline=True, fontsize=10);
	gridx = (xrange[1]-xrange[0])/Nx; gridy = (yrange[1]-yrange[0])/Ny;
	arrows = [];
	for i in np.arange(xrange[0], xrange[1], gridx):
		print(i);
		for j in np.arange(yrange[0]+0.01, yrange[1]+ 0.01, gridy):
			w = j; 
			mu11 = i;
			direct = w2_dir_wmu11(X, w, np.array([[mu11, mu12], mu2_fix]), ytruth, A, N);
			arrows.append(direct);
			length = np.linalg.norm(direct);
			direct = direct/length;
			plt.arrow(i, j, direct[0]/20, direct[1]/20, width=0.005, fc='blue', ec='blue', head_length = 0.03, head_width = 0.02);
	if (trace != ''):
		plt.plot(trace[:, 0], trace[:, 1], '*-', c='r', label = 'std GD trace', zorder = 5);
		plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
		plt.text(trace[0][0]-.5, trace[0][1]+0.1, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_{11}$: $x$ direction");
	plt.ylabel(r"$w$: $y$ direction");
	plt.legend(loc='upper left', bbox_to_anchor=(0, 1.5));
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300, bbox_inches = "tight");
	if (savpath2!=''):
		arrows=np.array(arrows);
		np.savetxt(savpath2, arrows);

##this function plots the w2 vector field and the convergence trace
##given the vector field as a 2d array 
##and the convergence trace as a 1d array
##Nx and Ny are the dimensions of the vf 2d array
def plot_w2VF_wmu11(X, mu2_fix, mu12, ytruth, vf, trace='', savpath='', xrange = [-3, 4], yrange = [0, 1], Nx=35, Ny=5):
	matlib.rcParams['text.usetex'] = True;
	dx = (xrange[1]-xrange[0])/45;
	dy = (yrange[1]-yrange[0])/45;
	x = np.arange(xrange[0],xrange[1],dx);
	y = np.arange(yrange[0],yrange[1],dy);
	plt.rcParams.update({'font.size': 30})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
		for j in range(y.size):
			w = yrange[0]+dy*j;
			mu11 = xrange[0]+dx*i;
			Z[i, j] = loss(X, 2, np.array([np.array([mu11, mu12]), mu2_fix]), np.array([w]), ytruth);
	Z=Z.T;
	fig, ax = plt.subplots(figsize=(35, 5));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/20);
	CS = ax.contour(P,Q,Z,levels = level, zorder = 0);
	ax.clabel(CS, inline=True, fontsize=10);
	gridx = (xrange[1]-xrange[0])/Nx; gridy = (yrange[1]-yrange[0])/Ny;
	##plot the vector field based on the 2d array vf
	for i in range(Nx):
		for j in range(Ny):
			index = Ny*i + j;
			direct = vf[index];
			length = np.linalg.norm(direct);
			direct = direct/length;
			plt.arrow(i*gridx + xrange[0], j*gridy + yrange[0], direct[0]/20, direct[1]/20, width=0.005, fc='blue', ec='blue', 
				head_length = 0.03, head_width = 0.02);
	##plot the trace
	if (trace != ''):
		plt.plot(trace[:, 0], trace[:, 1], '*-', c='r', label = r'$W^2$ natural GD trace', zorder = 5);
		plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
		plt.text(trace[0][0]-.5, trace[0][1]+0.1, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_{11}$: $x$ direction");
	plt.ylabel(r"$w$: $y$ direction");
	plt.legend(loc='upper left', bbox_to_anchor=(0, 1.5));
	if (savpath != ''):
		plt.savefig(savpath, bbox_inches='tight', dpi=300)



##this function completes standard gradient descent 
##the parameters involved are [mu11, w].
##initial is a 2-d vector of our initial parameter settings [mu11, w]
##lr is the step size, iteration is the number of steps we take
##savpath is the path to save the photo file of plot
##tracepath is the path to save the csv file that contains the trace
def stdGD_wmu11(X, mu2_fix, mu12, ytruth, initial, lr, iteration, savpath='', tracepath='', xrange = [-3, 4], yrange = [0, 1]):
	##first plot the loss function
	matlib.rcParams['text.usetex'] = True;
	dx = (xrange[1]-xrange[0])/45;
	dy = (yrange[1]-yrange[0])/45;
	x = np.arange(xrange[0],xrange[1],dx);
	y = np.arange(yrange[0],yrange[1],dy);
	plt.rcParams.update({'font.size': 30})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
		for j in range(y.size):
			w = yrange[0]+dy*j;
			mu11 = xrange[0]+dx*i;
			Z[i, j] = loss(X, 2, np.array([np.array([mu11, mu12]), mu2_fix]), np.array([w]), ytruth);
	Z=Z.T;
	fig, ax = plt.subplots(figsize=(35, 5));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/20);
	CS = ax.contour(P,Q,Z,levels = level, zorder = 0);
	ax.clabel(CS, inline=True, fontsize=10);
	##create an array to store the convergence parameters
	position = initial;
	trace = [initial];
	##do the std GD algorithm
	for i in range(iteration):
		direct, _ = gradient_w_mu11(X, position[1], np.array([[position[0], mu12], mu2_fix]), ytruth);
		position = position - lr*direct;
		trace.append(position);
	trace = np.array(trace);
	##plot the trace
	plt.plot(trace[:, 0], trace[:, 1], '*-', c='r', label = 'std GD trace', zorder = 5);
	plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
	plt.text(trace[0][0]-2, trace[0][1]+0.3, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_{11}$: $x$ direction");
	plt.ylabel(r"$w$: $y$ direction");
	plt.legend();
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300, bbox_inches = "tight");
	if (tracepath != ''):
		np.savetxt(tracepath, trace);
	return trace;


##this function completes l2 natural gradient descent 
##the parameters involved are [mu11, w].
##initial is a 2-d vector of our initial parameter settings [mu11, w]
##lr is the step size, iteration is the number of steps we take
##savpath is the path to save the photo file of plot
##tracepath is the path to save the csv file that contains the trace
def l2GD_wmu11(X, mu2_fix, mu12, ytruth, initial, lr, iteration, savpath='', tracepath='', xrange = [-3, 4], yrange = [0, 1]):
	##first plot the loss function
	matlib.rcParams['text.usetex'] = True;
	dx = (xrange[1]-xrange[0])/45;
	dy = (yrange[1]-yrange[0])/45;
	x = np.arange(xrange[0],xrange[1],dx);
	y = np.arange(yrange[0],yrange[1],dy);
	plt.rcParams.update({'font.size': 30})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
		for j in range(y.size):
			w = yrange[0]+dy*j;
			mu11 = xrange[0]+dx*i;
			Z[i, j] = loss(X, 2, np.array([np.array([mu11, mu12]), mu2_fix]), np.array([w]), ytruth);
	Z=Z.T;
	fig, ax = plt.subplots(figsize=(35, 5));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/20);
	CS = ax.contour(P,Q,Z,levels = level, zorder = 0);
	ax.clabel(CS, inline=True, fontsize=10);
	##create an array to store the convergence parameters
	position = initial;
	trace = [initial];
	##do the std GD algorithm
	for i in range(iteration):
		direct = l2_dir_wmu11(X, position[1], np.array([[position[0], mu12], mu2_fix]), ytruth);
		position = position + lr*direct;
		trace.append(position);
	trace = np.array(trace);
	##plot the trace
	plt.plot(trace[:, 0], trace[:, 1], '*-', c='r', label = r'$L^2$ natural GD trace', zorder = 5);
	plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
	plt.text(trace[0][0]-.5, trace[0][1]+0.1, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_{11}$: $x$ direction");
	plt.ylabel(r"$w$: $y$ direction");
	plt.legend();
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300, bbox_inches = "tight");
	if (tracepath != ''):
		np.savetxt(tracepath, trace);
	return trace;


##this function completes w2 natural gradient descent 
##the parameters involved are [mu11, w].
##initial is a 2-d vector of our initial parameter settings [mu11, w]
##lr is the step size, iteration is the number of steps we take
##savpath is the path to save the photo file of plot
##tracepath is the path to save the csv file that contains the trace
def w2GD_wmu11(X,A, N, mu2_fix, mu12, ytruth, initial, lr, iteration, savpath='', tracepath='', xrange = [-3, 4], yrange = [0, 1]):
	##first plot the loss function
	matlib.rcParams['text.usetex'] = True;
	dx = (xrange[1]-xrange[0])/45;
	dy = (yrange[1]-yrange[0])/45;
	x = np.arange(xrange[0],xrange[1],dx);
	y = np.arange(yrange[0],yrange[1],dy);
	plt.rcParams.update({'font.size': 30})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
		for j in range(y.size):
			w = yrange[0]+dy*j;
			mu11 = xrange[0]+dx*i;
			Z[i, j] = loss(X, 2, np.array([np.array([mu11, mu12]), mu2_fix]), np.array([w]), ytruth);
	Z=Z.T;
	fig, ax = plt.subplots(figsize=(35, 5));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/20);
	CS = ax.contour(P,Q,Z,levels = level, zorder = 0);
	ax.clabel(CS, inline=True, fontsize=10);
	##create an array to store the convergence parameters
	position = initial;
	trace = [initial];
	##do the std GD algorithm
	for i in range(iteration):
		print(i);
		direct = w2_dir_wmu11(X, position[1], np.array([[position[0], mu12], mu2_fix]), ytruth, A, N);
		position = position + lr*direct;
		trace.append(position);
	trace = np.array(trace);
	##plot the trace
	plt.plot(trace[:, 0], trace[:, 1], '*-', c='r', label = r'$W^2$ natural GD trace', zorder = 5);
	plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
	plt.text(trace[0][0]-.5, trace[0][1]+0.1, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_{11}$: $x$ direction");
	plt.ylabel(r"$w$: $y$ direction");
	plt.legend();
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300, bbox_inches = "tight");
	if (tracepath != ''):
		np.savetxt(tracepath, trace);
	return trace;



##N is the discretization number
##will save the arrows in the path if the path is not empty
def Vector_field_w2mu1(X, mu2_fix, w, ytruth, N, dx, dy, trace='', savpath='', arrowpath='',
						xrange = [-3.5, 8.5], yrange = [-3.5, 8.5]):
	A = discretize(N, dx, dy);
	matlib.rcParams['text.usetex'] = True;
	x = np.arange(xrange[0],xrange[1],grid);
	y = np.arange(yrange[0],yrange[1],grid);
	plt.rcParams.update({'font.size': 20})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
			for j in range(y.size):
				Z[i, j] = loss(X, 2, np.array([np.array([grid*i+xrange[0], grid*j+yrange[0]]), mu2_fix]),
							   np.array([w]), ytruth);
	fig, ax = plt.subplots(figsize=(10, 10));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/40);
	CS = ax.contour(Q,P,Z,levels = level, zorder = 0);
	ax.clabel(CS, inline=True, fontsize=10);
	arrows = [];
	for i in np.arange(xrange[0], xrange[1], 0.5):
		print(i);
		for j in np.arange(yrange[0]+0.01, yrange[1], 0.5):     
			mu1 = np.array([i, j]);
			direct = Natural_dir_mu1(X, N, np.array([w]), np.array([mu1, mu2_fix]), ytruth, A, 2);
			arrows.append(direct);
			length = np.linalg.norm(direct);
			direct = -direct/length;
			plt.arrow(i, j, direct[0]/3, direct[1]/3, width=0.01, fc='blue', ec='blue', head_length = 0.1, head_width = 0.05);
	if (trace != ''):
		plt.plot(trace[:, 0], trace[:, 1], '.-', c='r', label = r'$W_2$ Natural GD trace', zorder = 5);
		plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
		plt.text(trace[0][0]+.2, trace[0][1]+0.2, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_1$: $x$ direction");
	plt.ylabel(r"$\mu_1$: $y$ direction");
	plt.legend();
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300);
	if (arrowpath!=''):
		arrows = np.array(arrows);
		np.savetxt(arrowpath, arrows);

##this function plot the vector fields together with the convergence trace
##vfpath is the path of a csv files that contains all unnormalized directions of the arrowa
##we assume the vector field contains m by n arrows
##tracepath is the path to the csv file that records the convergence trace
def plot_VF_mu1(X, mu2_fix, w, ytruth, m=0, n=0, vfpath='', tracepath='', xrange=[-2, 7.5], yrange=[-2, 7.5], savpath=''):
	#first plot the loss function
	matlib.rcParams['text.usetex'] = True;
	x = np.arange(xrange[0],xrange[1],grid)
	y = np.arange(yrange[0],yrange[1],grid)
	plt.rcParams.update({'font.size': 20})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
			for j in range(y.size):
				Z[i, j] = loss(X, 2, np.array([np.array([grid*i+xrange[0], grid*j+yrange[0]]), mu2_fix]), np.array([w]), ytruth);
	fig, ax = plt.subplots(figsize=(10, 10));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/40);
	CS = ax.contour(Q,P,Z,levels = level, zorder = 0);
	ax.clabel(CS, inline=True, fontsize=10);
	##plot the vector field
	if vfpath != '':
		#read the file and get the vector field array
		vf = np.genfromtxt(vfpath);
		for i in range(m):
			for j in range(n):
				index = i*m + j;
				coord = np.array([xrange[0]+ 0.5*i, yrange[0] + 0.5*j]);
				direct = vf[index];
				direct = -direct/np.linalg.norm(direct);
				plt.arrow(coord[0], coord[1], direct[0]/3, direct[1]/3, width=0.01, fc='blue', ec='blue', head_length = 0.1, head_width = 0.05);
	##plot the convergence trace
	if tracepath != '':
		##read the trace file
		trace = np.genfromtxt(tracepath);
		plt.plot(trace[:, 0], trace[:, 1], '.-', c='r', label = r'$W_2$ Natural GD trace', zorder = 5);
		plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
		plt.text(trace[0][0]+.2, trace[0][1]+0.2, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_1$: $x$ direction");
	plt.ylabel(r"$\mu_1$: $y$ direction");
	plt.legend();
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300);
	return;

def Vector_field_l2mu1(X, mu2_fix, w, ytruth, N, trace='', savpath='', 
						xrange = [-3.5, 8.5], yrange = [-3.5, 8.5]):
	matlib.rcParams['text.usetex'] = True;
	plt.rcParams.update({'font.size': 20})
	x = np.arange(xrange[0],xrange[1],grid);
	y = np.arange(yrange[0],yrange[1],grid);
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
			for j in range(y.size):
				Z[i, j] = loss(X, 2, np.array([np.array([grid*i+xrange[0], grid*j+yrange[0]]), mu2_fix]), 
							  np.array([w]), ytruth);
	fig, ax = plt.subplots(figsize=(10, 10));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/40);
	CS = ax.contour(Q,P,Z,levels = level, zorder=0);
	ax.clabel(CS, inline=True, fontsize=10);
	for i in np.arange(xrange[0], xrange[1], 0.5):
		for j in np.arange(yrange[0], yrange[1], 0.5):
			mu1 = np.array([i, j]);
			direct = l2_Natural_dirFull_mu1(X, np.array([mu1, mu2_fix]), np.array([w]), N, ytruth);
			length = np.linalg.norm(direct);
			if (length<0.00000000001):
				plt.scatter(i, j, c = 'blue', label = 'global minimum', zorder = 20, s=100);
				continue;
			direct = direct/length;
			plt.arrow(i, j, direct[0]/3, direct[1]/3, 
					width=0.01, fc='blue', ec='blue', 
					  head_length = 0.1, head_width = 0.05);
	if (trace != ''):
		plt.plot(trace[:, 0], trace[:, 1], 
				 '.-', c='r', label = r'$L_2$ Natural GD trace', zorder = 5);
		plt.scatter(trace[0][0], 
				trace[0][1], s=150, c='g', marker=(5, 1), 
					zorder = 10, label = 'initial point')
		plt.text(trace[0][0]-1.3, 
				 trace[0][1]+0.2, 
				 '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_1$: $x$ direction");
	plt.ylabel(r"$\mu_1$: $y$ direction");
	plt.legend();
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300);

## xrange and yrange must be a 11 by 11 square
## w is a number, the weight
def Vector_field_stdmu2(X, mu1_fix, w, ytruth, trace='', savpath='', xrange = [-3.5, 8.5], yrange = [-3.5, 8.5]):
	matlib.rcParams['text.usetex'] = True;
	x = np.arange(xrange[0],xrange[1],grid);
	y = np.arange(yrange[0],yrange[1],grid);
	plt.rcParams.update({'font.size': 20})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
			for j in range(y.size):
				Z[i, j] = loss(X, 2, np.array([mu1_fix, np.array([grid*i+xrange[0], grid*j+yrange[0]])]), 
							   np.array([w]), ytruth);
	fig, ax = plt.subplots(figsize=(10, 10));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/40);
	CS = ax.contour(Q,P,Z,levels = level, zorder = 0);
	ax.clabel(CS, inline=True, fontsize=10);
	for i in np.arange(xrange[0], xrange[1], 0.5):
		print(i);
		for j in np.arange(yrange[0], yrange[1], 0.5):
			mu2 = np.array([i, j]);
			p, direct = gradient_mu2_all(X, np.array([w]), mu1_fix, mu2, ytruth);
			length = np.linalg.norm(direct);
			direct = -direct/length;
			plt.arrow(i, j, direct[0]/3, direct[1]/3, width=0.01, fc='blue', ec='blue', head_length = 0.1, head_width = 0.05);
	if (trace != ''):
		plt.plot(trace[:, 0], trace[:, 1], '*-', c='r', label = 'std GD trace', zorder = 5);
		plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
		plt.text(trace[0][0]-2, trace[0][1]+0.3, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_2$: $x$ direction");
	plt.ylabel(r"$\mu_2$: $y$ direction");
	plt.legend();
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300);

##N is the discretization number
def Vector_field_w2mu2(X, mu1_fix, w, ytruth, N, dx, dy, trace='', savpath='', arrowpath='',
						xrange = [-3.5, 8.5], yrange = [-3.5, 8.5]):
	A = discretize(N, dx, dy);
	matlib.rcParams['text.usetex'] = True;
	x = np.arange(xrange[0],xrange[1],grid);
	y = np.arange(yrange[0],yrange[1],grid);
	plt.rcParams.update({'font.size': 20})
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
			for j in range(y.size):
				Z[i, j] = loss(X, 2, np.array([mu1_fix, np.array([grid*i+xrange[0], grid*j+yrange[0]])]),
							   np.array([w]), ytruth);
	fig, ax = plt.subplots(figsize=(10, 10));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/40);
	CS = ax.contour(Q,P,Z,levels = level, zorder = 0);
	ax.clabel(CS, inline=True, fontsize=10);
	arrows = [];
	for i in np.arange(xrange[0], xrange[1], 0.5):
		print(i);
		for j in np.arange(yrange[0], yrange[1], 0.5):     
			mu2 = np.array([i, j]);
			direct = Natural_dir_mu2(X, N, np.array([w]), np.array([mu1_fix, mu2]), ytruth, A, 2); 
			arrows.append(direct);     
			length = np.linalg.norm(direct);
			direct = -direct/length;
			plt.arrow(i, j, direct[0]/3, direct[1]/3, width=0.01, fc='blue', ec='blue', head_length = 0.1, head_width = 0.05);
	if (trace != ''):
		plt.plot(trace[:, 0], trace[:, 1], '.-', c='r', label = r'$W_2$ Natural GD trace', zorder = 5);
		plt.scatter(trace[0][0], trace[0][1], s=150, c='g', marker=(5, 1), zorder = 10, label = 'initial point')
		plt.text(trace[0][0]+.2, trace[0][1]+0.2, '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_2$: $x$ direction");
	plt.ylabel(r"$\mu_2$: $y$ direction");
	plt.legend();
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300);
	if arrowpath != '':
		arrows = np.array(arrows);
		np.savetxt(arrowpath, arrows);
	

def Vector_field_l2mu2(X, mu1_fix, w, ytruth, N, trace='', savpath='', 
						xrange = [-3.5, 8.5], yrange = [-3.5, 8.5]):
	matlib.rcParams['text.usetex'] = True;
	plt.rcParams.update({'font.size': 20})
	x = np.arange(xrange[0],xrange[1],grid);
	y = np.arange(yrange[0],yrange[1],grid);
	P,Q = np.meshgrid(x,y)
	Z = np.zeros((x.size, y.size));
	for i in range(x.size):
			for j in range(y.size):
				Z[i, j] = loss(X, 2, np.array([mu1_fix, np.array([grid*i+xrange[0], grid*j+yrange[0]])]), 
							  np.array([w]), ytruth);
	fig, ax = plt.subplots(figsize=(10, 10));
	level = np.arange(Z.min(), Z.max(), (Z.max()-Z.min())/40);
	CS = ax.contour(Q,P,Z,levels = level, zorder=0);
	ax.clabel(CS, inline=True, fontsize=10);
	for i in np.arange(xrange[0], xrange[1], 0.5):
		for j in np.arange(yrange[0], yrange[1], 0.5):
			mu2 = np.array([i, j]);
			direct = l2_Natural_dirFull_mu2(X, np.array([mu1_fix, mu2]), np.array([w]), N, ytruth);
			length = np.linalg.norm(direct);
			direct = direct/length;
			plt.arrow(i, j, direct[0]/3, direct[1]/3, 
					width=0.01, fc='blue', ec='blue', 
					  head_length = 0.1, head_width = 0.05);
	if (trace != ''):
		plt.plot(trace[:, 0], trace[:, 1], 
				 '.-', c='r', label = r'$L_2$ Natural GD trace', zorder = 5);
		plt.scatter(trace[0][0], 
				trace[0][1], s=150, c='g', marker=(5, 1), 
					zorder = 10, label = 'initial point')
		plt.text(trace[0][0]-1.3, 
				 trace[0][1]+0.2, 
				 '({}, {})'.format(trace[0][0], trace[0][1]));
	plt.xlabel(r"$\mu_2$: $x$ direction");
	plt.ylabel(r"$\mu_2$: $y$ direction");
	plt.legend();
	if (savpath != ''):
		plt.savefig(savpath, dpi = 300);


#it returns both Y and Y.T Y
def Gw2_matrix(X, N, k, w, mu, ytruth, dx, dy):
	A = discretize(N, dx, dy);
	B = Bmatrix(X, N, w, mu, k, A);
	B = B.toarray();
	q, r= linalg.qr(B.T, mode = 'economic');
	gradp, gradu = U_grad(X, mu, w, k, N, ytruth);
	Z = np.linalg.solve(r.T, -gradp);
	Y = np.matmul(q, Z);
	return Y, Y.T@Y;

def plot_data(mu, w, xrange, yrange, grid, ang1, ang2, savepath = -1):
	fig, ax = plt.subplots(subplot_kw={"projection":"3d"});
	fig.set_figheight(10)
	fig.set_figwidth(15)
	X = np.arange(xrange[0], xrange[1], grid);
	Y = np.arange(yrange[0], yrange[1], grid);
	X, Y = np.meshgrid(X, Y);
	Z = np.zeros(X.shape)
	for i in range(Y.shape[0]):
		for j in range(X.shape[0]):
			x = X[j][j]; y = Y[i][i];
			Z[i, j] = model(np.array([x, y]), 2, mu, w);    
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
						   linewidth=0, antialiased=False)
	ax.view_init(ang1, ang2)
	ax.set_xlabel("x direction");
	ax.set_ylabel("y direction");
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_title("2 Mix of Guass: w: {} means: {}".format(w[0], mu));
	if (savepath!=-1):
		plt.savefig(savepath, dpi=300);
		



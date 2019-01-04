from __future__ import division

import optparse
import sys
import operator
import math
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn import neighbors
import numpy as np
from numpy import array
from sklearn import cross_validation
from textwrap import dedent
from sklearn import linear_model
from sklearn.cross_validation import KFold
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
import random
from itertools import groupby
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesClassifier
from random import shuffle
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.cluster import FeatureAgglomeration
from sklearn.externals.joblib import Memory
import tempfile
from sklearn.decomposition import PCA
from sklearn import cluster
import heapq
import time
import pickle
import scipy
from sklearn.tree import DecisionTreeClassifier
from scipy import special, stats
from scipy.sparse import issparse
from scipy import linalg
from mlpy import kmeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pylab as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import mean_squared_error

def norm(x):
	"""Compute the Euclidean or Frobenius norm of x.
	Returns the Euclidean norm when x is a vector, the Frobenius norm when x
	is a matrix (2-d array). More precise than sqrt(squared_norm(x)).
	"""
	x = np.asarray(x)
	nrm2, = linalg.get_blas_funcs(['nrm2'], [x])
	return nrm2(x)

def safe_sparse_dot(a, b, dense_output=False):
	"""Dot product that handle the sparse matrix case correctly
	Uses BLAS GEMM as replacement for numpy.dot where possible
	to avoid unnecessary copies.
	"""
	if issparse(a) or issparse(b):
		ret = a * b
		if dense_output and hasattr(ret, "toarray"):
			ret = ret.toarray()
		return ret
	else:
		return np.dot(a, b)

def f_regression2(X, y, center=True):
	"""Univariate linear regression tests
	Quick linear model for testing the effect of a single regressor,
	sequentially for many regressors.
	This is done in 3 steps:
	1. the regressor of interest and the data are orthogonalized
	wrt constant regressors
	2. the cross correlation between data and regressors is computed
	3. it is converted to an F score then to a p-value
	Parameters
	----------
	X : {array-like, sparse matrix} shape = (n_samples, n_features)
	The set of regressors that will tested sequentially.
	y : array of shape(n_samples).
	The data matrix
	center : True, bool,
	If true, X and y will be centered.
	Returns
	-------
	F : array, shape=(n_features,)
	F values of features.
	pval : array, shape=(n_features,)
	p-values of F-scores.
	if issparse(X) and center:
		raise ValueError("center=True only allowed for dense data")
	#X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=np.float)
	if center:
		y = y - np.mean(y)
		X = X.copy('F') # faster in fortran
		X -= X.mean(axis=0)
	# compute the correlation
	corr = safe_sparse_dot(y, X)
	# XXX could use corr /= row_norms(X.T) here, but the test doesn't pass
	corr /= np.asarray(np.sqrt(np.square(X).sum(axis=0))).ravel()
	corr /= norm(y)
	# convert to p-value
	degrees_of_freedom = y.size - (2 if center else 1)
	F = corr ** 2 / (1 - corr ** 2) * degrees_of_freedom
	pv = stats.f.sf(F, 1, degrees_of_freedom)
	"""
	F=[]
	pv=[]
	for x1 in range(len(X[0])):
		'''
		F.append(spearmanr(X[:,x1],y)[0])	#abs
		pv.append(spearmanr(X[:,x1],y)[1])
		#stats.ks_2samp(X[:,x1],y)	stats.ttest_ind(X[:,x1],y)	scipy.stats.f_oneway(X[:,x1],y)	stats.ttest_rel(X[:,x1],y)	 stats.wilcoxon(x, y=None, zero_method='wilcox', correction=False)[source]	 scipy.stats.kruskal(X[:,x1],y)	spearmanr(X[:,x1],y)
		'''
		cor=spearmanr(X[:,x1],y)
		F.append(abs(cor[0]))
		pv.append(cor[1])
	F=np.array(F)
	pv=np.array(pv)
	return F, pv


def is_numeric(x):
    """Returns whether the given string can be interpreted as a number."""
    try:
        float(x)
        return True
    except:
        return False


def addslashes(s):
    d = {'"':'\\"', "'":"\\'", "\0":"\\\0", "\\":"\\\\"}
    return ''.join(d.get(c, c) for c in s)


def read_distanceMatrix(fname):
    #t_exp={}
    distance=[]
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
	row=[]
        el=line.split(' ')
	for e in el:
		if e!="":
			row.append(abs(float(e)))
	distance.append(row)
    	
    return distance

def read_unip(fname):
    #t_exp={}
    mapping={}
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        genes=line.split('\t')
        x=0
        #for g in genes:
        #    if x>0:
        #       mapping[g]=genes[0]
        #    x+=1
        #print genes[1],
        #print genes[0]
        mapping[genes[0]]=genes[1]
    
    return mapping

def read_unip(fname):
    #t_exp={}
    mapping={}
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        genes=line.split('\t')
        x=0
        #for g in genes:
        #    if x>0:
        #       mapping[g]=genes[0]
        #    x+=1
        #print genes[1],
        #print genes[0]
        mapping[genes[0]]=genes[1]
    
    return mapping

def read_tsg(fname):
    #t_exp={}
    tsg=[]
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        genes=line.split('\t')
        tsg.append(genes[1])
    
    return tsg

def read_compl(fname, mapping, compl2):
    #t_exp={}
    compl={}
    x=1
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        prots=line.split('\t')
        row = []
        for p in prots:
            if p in mapping:
               compl2[mapping[p]]=x
               row.append(mapping[p])
        compl[x]=row
        #    if x>0:
        #       mapping[g]=genes[0]
        
        x+=1
        #print genes[1],
        #print genes[0]
    
    return compl   

def read_compl_c1(fname, mapping, compl2):
    #t_exp={}
    compl={}
    x=1
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        prots=line.split(' ')
        row = []
        for p in prots:
            compl2[p]=x
            row.append(p)
        compl[x]=row
        #    if x>0:
        #       mapping[g]=genes[0]
        
        x+=1
        #print genes[1],
        #print genes[0]
    
    return compl

def read_int(fname, mapping):
    #t_exp={}
    inter={}
    x=1
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        prots=line.split('\t')
        if prots[0] in inter:
           #inter[prots[0]]+=1
           inter[prots[0]].append(prots[1])
        else:
             #inter[prots[0]]=1
             inter[prots[0]]=[prots[1]]
        if prots[1] in inter:
           #inter[prots[1]]+=1
           inter[prots[1]].append(prots[0])
        else:
             #inter[prots[1]]=1
             inter[prots[1]]=[prots[0]]
        #print genes[1],
        #print genes[0]
    
    return inter

def read_texp(fname):
    #t_exp={}
    mapping={}
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        genes=line.split('\t')
        x=0
        #for g in genes:
        #    if x>0:
        #       mapping[g]=genes[0]
        #    x+=1
        #print genes[1],
        #print genes[0]
        mapping[genes[0]]=genes[1]
    
    return mapping
    
    
def read_ccel(fname):
    #t_exp={}
    ccel={}
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        line=line.replace('"', '')
        words=line.split(',')

        row=[words[3],words[4],words[5],words[6]] #gender site histology histology-sub
        ccel[words[0]]=row
        ccel[words[1]]=row

    return ccel

def count_mutations(fname):
    #t_exp={}
    count={}
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        line=line.replace('"', '')
        words=line.split('\t')
        ccel=words[2].replace('-', '')
	'''
        if ccel in count:
           if words[0] in count[ccel]:
              count[ccel][words[0]]+=1
              #print ccel+"-c-"+words[0]
           else:
                count[ccel][words[0]]=1
                #print ccel+"-b-"+words[0]
        else:
             count[ccel] = {}
             count[ccel][words[0]]=1
	'''
	
        if words[0] in count:
        	count[words[0]]+=1
        else:
             	count[words[0]] = 1


    return count
    
def mutations_ccel(fname, types):
    #t_exp={}
    genes=[]	
    mutationCCEL={}
    mutationSITE={}
    mutationSITE2={}
    mutationHIST={}
    mutationHISTS={}
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        line=line.replace('"', '')
        words=line.split('\t')
        ccel=words[2].replace('-', '')
	if ccel in types:
		if not words[0] in genes:
			genes.append(words[0])
		if ccel in mutationCCEL:
			mutationCCEL[ccel].append(words[0])
		else:
			mutationCCEL[ccel]=[]
			mutationCCEL[ccel].append(words[0])

		if ccel in mutationSITE:
			mutationSITE[ccel].append(words[0])
		else:
			mutationSITE[ccel]=[]
			mutationSITE[ccel].append(words[0])

		if ccel in mutationSITE2:
			mutationSITE[ccel].append(words[0])
		else:
			mutationSITE[ccel]=[]
			mutationSITE[ccel].append(words[0])

		if ccel in mutationHIST:
			mutationHIST[ccel].append(words[0])
		else:
			mutationHIST[ccel]=[]
			mutationHIST[ccel].append(words[0])

		if ccel in mutationHISTS:
			mutationHISTS[ccel].append(words[0])
		else:
			mutationHISTS[ccel]=[]
			mutationHISTS[ccel].append(words[0])

    return genes,mutationCCEL,mutationSITE,mutationSITE2,mutationHIST,mutationHISTS

def count_smutations(fname):
    #t_exp={}
    count={}
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        line=line.replace('"', '')
        words=line.split('\t')
        ccel=words[0].replace('-', '')

        if ccel in count:
           count[ccel]+=1
              #print ccel+"-c-"+words[0]
        else:
             count[ccel]=1
                #print ccel+"-b-"+words[0]
             #print ccel+"-a-"+words[0]
        #print ccel+" "+str(count[ccel])


    return count
    
def cancerGenes(fname):
    #t_exp={}
    genes=[]
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        words=line.split('\t')
        genes.append(words[1])


    return genes

def cancerGenes2(fname, types,hist,site,site2):
    #t_exp={}
    genes=[]
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        words=line.split('\t')
	ty=words[23].split(',')
	#print ty
	for t in ty: 
		if t in types or t in hist or t in site or t in site2:
			if not words[1] in genes:
        			genes.append(words[1])


    return genes

def cancerGenes_site(fname):
    #t_exp={}
    genes={}
	
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        words=line.split('\t')
	ty=words[23].split(',')
	#print ty
	for t in ty: 
		if t in genes:
			genes[t].append(words[1])
		else:
			genes[t]=[]
			genes[t].append(words[1])


    return genes
    
def essentialGenes(fname):
    #t_exp={}
    genes=[]
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        words=line.split('\t')
        genes.append(words[0])


    return genes

def readTF(fname):
    #t_exp={}
    tfs={}
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        words=line.split(';')
	if words[2] in tfs:
		if not words[4] in tfs[words[2]]:
			tfs[words[2]].append(words[4])
	else:
		tfs[words[2]]=[words[4]]

    return tfs

def read_texp(fname, fname2):
    t_exp={}
    mapping={}
    gene=[]
    with open(fname2) as f:
	    for line in f.readlines():
		line=line.replace("\n", "")
		line=line.replace("\r", "")
		genes=line.split('\t')
		x=0
		for g in genes:
		    if x>0:
		       mapping[g]=genes[0]
		    x+=1
		#print genes[1],
		#print genes[0]
		#mapping[genes[1]]=genes[0]


    nline=0
    with open(fname) as f:
	    for line in f.readlines():
		if nline>0:
		  line=line.replace("\n", "")
		  line=line.replace("\r", "")
		  genes=line.split('\t')
		  x=0
		  if genes[0] in mapping:		
			  gene.append(mapping[genes[0]])
			  for t in genes:
			      if x>0:
				 if t in t_exp:
				 	t_exp[t].append(mapping[genes[0]])
				 else:
				 	t_exp[t]=[]	
				 	t_exp[t].append(mapping[genes[0]])			
			      x+=1
		  #print mapping[genes[0]]
		  #if genes[0] in mapping:
		  #   t_exp[mapping[genes[0]]]=tissues
		nline+=1
	    #print t_exp
    return gene,t_exp

def read_miRNA2(fname):
    #t_exp={}
    genes=[]
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        words=line.split('\t')
        genes.append(words[6])


    return genes
def read_miRNA(fname):
    #t_exp={}
    genes={}
	
    for line in open(fname):
        line=line.replace("\n", "")
        line=line.replace("\r", "")
        words=line.split('\t')
	ty=words[13].split(',')
	#print ty
	for t in ty: 
		if t in genes:
			genes[t].append(words[6])
		else:
			genes[t]=[]
			genes[t].append(words[6])
    return genes

def gep(argv):
    #read gene and ccel info
    nexp_genes, t_exp = read_texp(sys.argv[1],sys.argv[2])
    unip2gene = read_unip(sys.argv[3])
    unip2gene_int = read_unip(sys.argv[3])
    compl2 = {}
    ccel_ann = read_ccel(sys.argv[4])
    mutations = count_mutations(sys.argv[5])
    smutations = count_smutations(sys.argv[6])
    cancer_genes = cancerGenes(sys.argv[7])
    oncogenesBySite = cancerGenes_site(sys.argv[7])
    tsg = []
    compl = read_compl(sys.argv[8],unip2gene,compl2)
    compl2_c1={}
    compl_c1 = read_compl_c1(sys.argv[9],unip2gene,compl2_c1)
    interactions = read_int(sys.argv[10],unip2gene_int)
    ppi=read_int(sys.argv[11],unip2gene_int)
    tsg=read_tsg(sys.argv[13])
    essG=essentialGenes(sys.argv[14])
    essG=list(set(essG))
    driverGenes=essentialGenes(sys.argv[15])
    priority=essentialGenes(sys.argv[16])
    tfs=readTF(sys.argv[17])
    top_compl=essentialGenes(sys.argv[18])
    mirnas = read_miRNA2(sys.argv[21])
    miRNABySite=read_miRNA(sys.argv[21])
    distanceMatrix=read_distanceMatrix(sys.argv[22])

    #load challenge data
    with open('obj/expr.pkl', 'rb') as handle:
    	expr = pickle.load(handle)
    with open('obj/p2g.pkl', 'rb') as handle:
    	p2g = pickle.load(handle)
    with open('obj/expr1.pkl', 'rb') as handle:
    	expr1 = pickle.load(handle)
    with open('obj/expr2b.pkl', 'rb') as handle:
    	expr2b = pickle.load(handle)
    with open('obj/expr2.pkl', 'rb') as handle:
    	expr2 = pickle.load(handle)
    with open('obj/cnv.pkl', 'rb') as handle:
    	cnv = pickle.load(handle)
    with open('obj/cnv_g.pkl', 'rb') as handle:
    	cnv_g = pickle.load(handle)
    with open('obj/cnv2.pkl', 'rb') as handle:
    	cnv2 = pickle.load(handle)
    with open('obj/genes.pkl', 'rb') as handle:
    	genes = pickle.load(handle)
    with open('obj/headers.pkl', 'rb') as handle:
    	headers = pickle.load(handle)
    with open('obj/headers_t.pkl', 'rb') as handle:
    	headers_t = pickle.load(handle)
    with open('obj/es.pkl', 'rb') as handle:
    	es = pickle.load(handle)
    with open('obj/essent2.pkl', 'rb') as handle:
    	essent2 = pickle.load(handle)
    with open('obj/essent3.pkl', 'rb') as handle:
    	essent3 = pickle.load(handle)
    with open('obj/geneInEssent.pkl', 'rb') as handle:
    	geneInEssent = pickle.load(handle)

    with open('obj/mut.pkl', 'rb') as handle:
    	mut = pickle.load(handle)
    with open('obj/mut2.pkl', 'rb') as handle:
    	mut2 = pickle.load(handle)
    with open('obj/mut3.pkl', 'rb') as handle:
    	mut3 = pickle.load(handle)
    with open('obj/mut_test.pkl', 'rb') as handle:
    	mut_test = pickle.load(handle)
    with open('obj/mut2_test.pkl', 'rb') as handle:
    	mut2_test = pickle.load(handle)
    with open('obj/mut3_test.pkl', 'rb') as handle:
    	mut3_test = pickle.load(handle)

    print mut3.keys()
    print mut3_test.keys()

    #join ccel data
    id_type = []
    id_site = []
    id_site2 = []
    id_hist = []
    id_hists = []

    ccel_type = {}
    ccel_site = {}
    ccel_site2 = {}
    ccel_gender = {}
    ccel_hist = {}
    ccel_hists = {}
    nline=0

    tot_ccel=[]	
    with open(sys.argv[12]) as f:
	    for line in f.readlines():
		line=line.replace("\n", "")
		line=line.replace("\r", "")

		ccel=line.split('\t')
		tot_ccel.append(ccel[0])
		if not nline==0:
		   ccel_type[ccel[0]]=ccel[2]
		   ccel_site[ccel[0]]=ccel[3]
		   found=False
		   if ccel[0] in ccel_ann:
		      found=True
		      ccel_gender[ccel[0]]=ccel_ann[ccel[0]][0]
		      ccel_site2[ccel[0]]=ccel_ann[ccel[0]][1]
		      ccel_hist[ccel[0]]=ccel_ann[ccel[0]][2]
		      ccel_hists[ccel[0]]=ccel_ann[ccel[0]][3]
		   if ccel[1] in ccel_ann and not found:
		      found=True
		      ccel_gender[ccel[0]]=ccel_ann[ccel[1]][0]
		      ccel_site2[ccel[0]]=ccel_ann[ccel[1]][1]
		      ccel_hist[ccel[0]]=ccel_ann[ccel[1]][2]
		      ccel_hists[ccel[0]]=ccel_ann[ccel[1]][3]
		   if not found:
		      print "MISSING!"+ccel[0]
		      ccel_gender[ccel[0]]=''
		      ccel_site2[ccel[0]]=''
		      ccel_hist[ccel[0]]=''
		      ccel_hists[ccel[0]]=''

		nline+=1
    for s in ccel_site:
        if not (ccel_site[s] in id_site):
           id_site.append(ccel_site[s])

    for s in ccel_type:
        if not (ccel_type[s] in id_type):
           id_type.append(ccel_type[s])

    for s in ccel_site2:
        if not (ccel_site2[s] in id_site2):
           id_site2.append(ccel_site2[s])

    for s in ccel_hist:
        if not (ccel_hist[s] in id_hist):
           id_hist.append(ccel_hist[s])

    for s in ccel_hists:
        if not (ccel_hists[s] in id_hists):
           id_hists.append(ccel_hists[s])


    mutatedGenes,mutationCCEL,mutationSITE,mutationSITE2,mutationHIST,mutationHISTS = mutations_ccel(sys.argv[5],headers)

    #vectorize ccel data
    c_features={}
    for c in tot_ccel:
        if c!="Name" and c!="Description":
           c_features[c]=[]
           #row.append(smutations[c])
           tissID=0
           for id1,t in enumerate(id_type):
               if t==str(ccel_type[c]):
                  c_features[c].append(1)
                           #tissID=id1
               else:
                    c_features[c].append(0)

               #row.append(tissID)
               #     siteID=0
               for id1,s in enumerate(id_site):
                   if s==str(ccel_site[c]):
                      c_features[c].append(1)
                            #siteID=id1
                   else:
                        c_features[c].append(0)
                    #row.append(siteID)
               gender=0
               if ccel_gender[c]=='M':
                  gender=1
               if ccel_gender[c]=='F':
                  gender=2
               c_features[c].append(gender)
                    #siteID=0
                    

               for id1,s in enumerate(id_site2):
                   if s==str(ccel_site2[c]):
                      c_features[c].append(1)
                   else:
                        c_features[c].append(0)
                            #siteID=id1
                    #row.append(siteID)
               #siteID=0
               for id1,s in enumerate(id_hist):
                   if s==str(ccel_hist[c]):
                      c_features[c].append(1)
                   else:
                        c_features[c].append(0)
                            #siteID=id1
                    #row.append(siteID)
               #siteID=0
               for id1,s in enumerate(id_hists):
                   if s==str(ccel_hists[c]):
                      c_features[c].append(1)
                   else:
                        c_features[c].append(0)
                            #siteID=id1

    #get not noise genes
    notNoise=[]
    std={}
    tot=0
    for c in headers:
	if c!="Name" and c!="Description":
		es=np.array(essent3[c])
		avg=np.mean(es)
		std[c]=np.std(es)
    #print len(headers)
    #print len(headers)
	
    for g in geneInEssent:
	es=np.array(essent2[g])
	tot+=1
	count=0
	for i,e in enumerate(es):
		if abs(float(e))>3*std[headers[i+2]]:
			count+=1
	if count>20:
		notNoise.append(g)
    print len(notNoise)
    #put data in the right format for lerner
    intr=0
    tot_sp1=0
    tot_sp2=0
    s1_prediction={}
    full=[]
    full_exp=[]
    full_cnv=[]
    full_ccel=[]
    full_s=[]
    full2=[]
    full3=[]
    full4=[]
    full3_test=[]
    full4_test=[]
    full_rank=[]
    minmax_fe=[]
    minmax_fc=[]
    onco_features=[]
    ccel_features=[]
    mut_features=[]
    expression_features=[]
    other_features=[]
    for c in headers:
        if c!="Name" and c!="Description":
           row=[]
           row2=[]
           drow={}
           drow2={}
           #row=row+c_features[c]
           for ex in genes:
               #row.append((expr1[c+ex]-exp_avg)/exp_std)
               row.append(expr1[c+ex])
               drow[ex]=expr1[c+ex]
           for ex in cnv_g:
               row2.append(float(cnv[c+ex]))
               drow2[ex]=cnv[c+ex]
           	
           '''	
           sorted_x = sorted(drow.iteritems(), key=operator.itemgetter(1))
           etop=sorted_x[0:10]
           ebot=sorted_x[-10:]
           for e in etop:
		minmax_fe.append(e[0])
           for e in ebot:
		minmax_fe.append(e[0])
           sorted_x = sorted(drow2.iteritems(), key=operator.itemgetter(1))
           etop=sorted_x[0:10]
           ebot=sorted_x[-10:]
           for e in etop:
		minmax_fc.append(e[0])
           for e in ebot:
		minmax_fc.append(e[0])
           '''	
           full_exp.append(row)
           full_cnv.append(row2)
           full_ccel.append(c_features[c])
           mutRow=[]	
           for g in mutatedGenes:
		if c in mutationCCEL:
			if g in mutationCCEL[c]:
				mutRow.append(1)
			else:
				mutRow.append(0)
		else:
			mutRow.append(0)
           	
           oncoRow=[]		
           print c
           for g in cancer_genes:
		if ccel_site[c] in oncogenesBySite:
			if g in oncogenesBySite[ccel_site[c]]:
				oncoRow.append(1)
			else:
				oncoRow.append(0)
		elif ccel_site2[c] in oncogenesBySite:
			if g in oncogenesBySite[ccel_site2[c]]:
				oncoRow.append(1)
			else:
				oncoRow.append(0)
		else:
			oncoRow.append(0)

		
           nexp=[]	
           for g in nexp_genes:
		if ccel_site[c] in t_exp:
			if g in t_exp[ccel_site[c]]:
				nexp.append(1)
			else:
				nexp.append(0)
		elif ccel_site2[c] in t_exp:
			if g in t_exp[ccel_site2[c]]:
				nexp.append(1)
			else:
				nexp.append(0)
		else:
			nexp.append(0)
	
           miRNA=[]		
           print c
           for g in mirnas:
		if ccel_site[c] in miRNABySite:
			if g in miRNABySite[ccel_site[c]]:
				miRNA.append(1)
			else:
				miRNA.append(0)
		elif ccel_site2[c] in miRNABySite:
			if g in miRNABySite[ccel_site2[c]]:
				miRNA.append(1)
			else:
				miRNA.append(0)
		else:
			miRNA.append(0)

           onco_features.append(oncoRow)       
           ccel_features.append(c_features[c])  
           mut_features.append(mutRow)           
           expression_features.append(nexp)
           full3.append(row+row2+c_features[c]+oncoRow+nexp)
           other_features.append(c_features[c]+oncoRow+nexp)
           #print stats.rankdata(np.array(row)).tolist()		
           full4.append(stats.rankdata(np.array(row)).tolist()+stats.rankdata(np.array(row2)).tolist()+c_features[c]+oncoRow+nexp+mut3[c])	

           #print len(row)

    print len(full4)
    print len(full4[0])   
    #full_exp=preprocessing.binarize(np.array(full_exp),threshold=np.mean(full_exp)).tolist()
    #full_cnv=preprocessing.binarize(np.array(full_cnv),threshold=np.mean(full_cnv)).tolist()
    full_exp=np.array(full_exp)
    for i,c in enumerate(full_exp):
	#print stats.rankdata(full5[:,i])
	#print len(full_exp[:,i])
	full_exp[i]=stats.rankdata(full_exp[i])
    full_cnv=np.array(full_cnv)
    for i,c in enumerate(full_cnv):
	#print stats.rankdata(full5[:,i])
	#print len(full_cnv[:,i])
	full_cnv[i]=stats.rankdata(full_cnv[i])

    full_exp=preprocessing.scale(full_exp)
    full_cnv=preprocessing.scale(full_cnv)

    full5=np.concatenate((full_exp,full_cnv),axis=1)
    full5=np.concatenate((full5,np.array(full3)),axis=1)

    print full5
    print len(full5)
    print len(full5[0])	
    full5=full5.tolist()

    #print full5	
    full_s=[]
    full_s2=[]
    for c in headers:
        if c!="Name" and c!="Description":
           row=[]
           for ex in genes:
               row.append(expr1[c+ex])
           full_s.append(row)
    for c in headers_t:
        if c!="Name" and c!="Description":
           row=[]
           for ex in genes:
               row.append(expr2[c+ex])
           full_s2.append(row)
    print len(full_s)
    print len(full_s2)
    #full_s=np.concatenate((np.array(full_s),np.array(full_s2)))
    full_s=np.array(full_s)
    print len(full_s[0])
    expr_c={}
    lbs=[]
    for c in headers:
        if c!="Name" and c!="Description":
		expr_c[c]=[]
		lbs.append(c+" - "+ccel_type[c])
		for ex in genes:
			expr_c[c].append(expr1[str(c+ex)])

    #distanceMatrix=np.zeros(shape=(len(expr_c),len(expr_c)))	
    '''	
    for i1,c in enumerate(expr_c):
	for i2,c2 in enumerate(expr_c):
		distanceMatrix[i1, i2]=pearsonr(np.array(expr_c[c]),np.array(expr_c[c2]))[0]
    '''
    r=dendrogram(linkage(distanceMatrix, method='complete'), color_threshold=0.3, leaf_font_size=7)
    #plt.show()
    leaves=r['leaves']

    '''
    print leaves
    distanceMatrix=np.zeros(shape=(len(essent3)-2,len(essent3)-2))	
    #sel = VarianceThreshold(threshold=0.1)
    #expr_c=np.array(expr_c.values())
    #vs=sel.fit(expr_c)
    #expr_c=vs.transform(expr_c)	
    for i1,c in enumerate(essent3):
        if c!="Name" and c!="Description":
		for i2,c2 in enumerate(essent3):
        		if c2!="Name" and c2!="Description":
				distanceMatrix[i1-2, i2-2]=pearsonr(np.array(essent3[c]),np.array(essent3[c2]))[0]
    

    #r=dendrogram(linkage(distanceMatrix, method='complete'), color_threshold=0.3, leaf_font_size=7, labels=lbs)

    #print distanceMatrix
    r=dendrogram(linkage(distanceMatrix, method='complete'), color_threshold=0.3, leaf_font_size=7)
    plt.show()	

    leaves_e=r['leaves']
    print leaves_e
    dist_e=np.zeros(shape=(len(expr_c),len(expr_c)))	

    dist=np.zeros(shape=(len(expr_c),len(expr_c)))	

    for i,c in enumerate(expr_c):
	for i2,c2 in enumerate(expr_c):
		dist_e[leaves_e[i],leaves_e[i2]]=abs(i-i2)
		dist_e[leaves_e[i2],leaves_e[i]]=abs(i-i2)
    for i,c in enumerate(expr_c):
	for i2,c2 in enumerate(expr_c):
		dist[leaves[i],leaves[i2]]=abs(i-i2)
		dist[leaves[i2],leaves[i]]=abs(i-i2)

    avg_corr=0
    print dist
    print dist_e
    for i,c in enumerate(expr_c):
	avg_corr+=spearmanr(np.array(dist[i]),np.array(dist_e[i]))[0]
    print (avg_corr/len(essent3))
    '''

    clusters={}
    '''
    clusters[0]=leaves[0:20]
    clusters[1]=leaves[20:32]
    clusters[2]=leaves[32:44]
    clusters[3]=leaves[44:48]
    clusters[4]=leaves[48:51]
    clusters[5]=leaves[51:65]
    clusters[6]=leaves[65:78]
    clusters[7]=leaves[78:91]
    clusters[8]=leaves[91:]
    '''

    clusters[0]=leaves[0:5]
    clusters[1]=leaves[5:23]
    clusters[2]=leaves[23:42]
    clusters[3]=leaves[42:49]
    clusters[4]=leaves[49:69]
    clusters[5]=leaves[69:85]
    clusters[6]=leaves[85:111]
    clusters[7]=leaves[111:132]
    clusters[8]=leaves[132:139]
    clusters[9]=leaves[139:]
    print clusters
    print leaves	
    #plt.show()
    #swap essentialities
    essent_s={}
    distance=np.zeros(shape=(len(essent3),len(essent3)))	
    for i1,c in enumerate(essent3):
        if c!="Name" and c!="Description":
		m=0
		for i2,c2 in enumerate(essent3):
        		if c2!="Name" and c2!="Description":
				if i1!=i2:
					if pearsonr(np.array(essent3[c]),np.array(essent3[c2]))[0]>m:
						essent_s[c]=essent3[c]
					#distance[i1,i2]=pearsonr(np.array(essent3[c]),np.array(essent3[c2]))[0]
					
		#m=max(distance[i1,:])
		#for i,d in enumerate(distance[i1,:]):
		#	if d==m:
		#		essent_s[c]=list(essent3[c1])
    essent_s=np.array(essent_s.values()).T
    essent_s2={}
    for i,g in enumerate(geneInEssent):
	essent_s2[g]=essent_s[i].tolist()
				
		
    #f = gcf()
    #plt.show()
    for c in headers_t:
        if c!="Name" and c!="Description":
           row2=[]
           row=[]
           for ex in genes:
               row.append(expr2[c+ex])
           for ex in cnv_g:
               row2.append(float(cnv2[c+ex]))

           mutRow=[]	
           for g in mutatedGenes:
		if c in mutationCCEL:
			if g in mutationCCEL[c]:
				mutRow.append(1)
			else:
				mutRow.append(0)
		else:
			mutRow.append(0)
           	
           oncoRow=[]		
           print c
           for g in cancer_genes:
		if ccel_site[c] in oncogenesBySite:
			if g in oncogenesBySite[ccel_site[c]]:
				oncoRow.append(1)
			else:
				oncoRow.append(0)
		elif ccel_site2[c] in oncogenesBySite:
			if g in oncogenesBySite[ccel_site2[c]]:
				oncoRow.append(1)
			else:
				oncoRow.append(0)
		else:
			oncoRow.append(0)

		
           nexp=[]	
           for g in nexp_genes:
		if ccel_site[c] in t_exp:
			if g in t_exp[ccel_site[c]]:
				nexp.append(1)
			else:
				nexp.append(0)
		elif ccel_site2[c] in t_exp:
			if g in t_exp[ccel_site2[c]]:
				nexp.append(1)
			else:
				nexp.append(0)
		else:
			nexp.append(0)

           miRNA=[]		
           print c
           for g in mirnas:
		if ccel_site[c] in miRNABySite:
			if g in miRNABySite[ccel_site[c]]:
				miRNA.append(1)
			else:
				miRNA.append(0)
		elif ccel_site2[c] in miRNABySite:
			if g in miRNABySite[ccel_site2[c]]:
				miRNA.append(1)
			else:
				miRNA.append(0)
		else:
			miRNA.append(0)

	
           full3_test.append(row+row2+c_features[c]+oncoRow+nexp)
           full4_test.append(row+row2+c_features[c]+oncoRow+nexp+mut3_test[c])
           row2=row+row2+c_features[c]
           full2.append(row2)
    #xt=np.array(full)
    #test=np.array(full2)



    sel = VarianceThreshold(threshold=0)
    full3=np.array(full3)
    full3_test=np.array(full3_test)
    tmp=np.concatenate((full3,full3_test),axis=0)
    full3=np.concatenate((full3,full3_test),axis=0)
    full3=full3.tolist()
    vs=sel.fit(tmp)
    full3=vs.transform(full3)
    full3=full3.tolist()
    full3_test=vs.transform(full3_test)
    full3_test=full3_test.tolist()   

    full4=np.array(full4)
    full4_test=np.array(full4_test)
    tmp=np.concatenate((full4,full4_test),axis=0)
    vs=sel.fit(tmp)
    full4=vs.transform(full4)
    full4=full4.tolist()
    full4_test=vs.transform(full4_test)
    full4_test=full4_test.tolist()
    inExp=0
    ninExp=0

    c_feat2=[]
    for c in headers:
	if c!="Name" and c!="Description":
		row=list(c_features[c])
		c_feat2.append(row)

    #feature selection strategies
    features={}
    alias=np.array(genes+cnv_g)
    #features['oncogenes']=[]
    #features['mutations']=[]
    #features['tsg']=[]
    #features['essential']=[]
    features['control']=[]
    #features['driver']=[]
    #features['top_compl']=[]
    #features['minmax']=[]
    #features['tf']=[]
    features_test={}
    features_test['oncogenes']=[]
    features_test['mutations']=[]
    features_test['tsg']=[]
    features_test['essential']=[]
    features_test['control']=[]
    features_test['driver']=[]
    '''
    for tf in tfs:
	tmp=[]
    	for i,c in enumerate(headers):
    		if (i-2)>=0:	
        		exp_list=[]
                      	for p in tfs[tf]:
                          	if (str(c)+str(p)) in expr:
                             		exp_list.append(float(expr[str(c)+str(p)]))
                          	if (str(c)+str(p)) in cnv:
                             	mlpy.kmeans(x, k=3, plus=True)	exp_list.append(float(cnv[str(c)+str(p)]))
			if len(exp_list)>2:
               			tmp.append(np.array(exp_list))
	if len(tmp)>1:
		features[tf]=tmp
		print str(tf)+": "+str(len(tmp[0]))
    for cc in top_compl:
	tmp=[]
    	for i,c in enumerate(headers):
    		if (i-2)>=0:	
        		exp_list=[]
                      	for p in compl[cc]:
                          	if (str(c)+str(p)) in expr:
                             		exp_list.append(float(expr[str(c)+str(p)]))
                          	if (str(c)+str(p)) in cnv:
                             		exp_list.append(float(cnv[str(c)+str(p)]))
			if len(exp_list)>2:
               			tmp.append(np.array(exp_list))
	if len(tmp)>1:
		features[cc]=tmp
    for cc in compl:
	tmp=[]
    	for i,c in enumerate(headers):
    		if (i-2)>=0:	
        		exp_list=[]
                      	for p in compl[cc]:
                          	if (str(c)+str(p)) in expr:
                             		exp_list.append(float(expr[str(c)+str(p)]))
                          	if (str(c)+str(p)) in cnv:
                             		exp_list.append(float(cnv[str(c)+str(p)]))
			if len(exp_list)>2:
               			tmp.append(np.array(exp_list))
	if len(tmp)>1:
		features[cc]=tmp
    if g in ppi:
	for p in ppi[g]:
		if ((str(c)+str(p)) in expr) and (not (p==g)):
         		exp_list.append(float(expr[str(c)+str(p)]))
    if (str(c)+str(g) in expr):
	exp_list.append(float(expr[str(c)+str(g)]))
    '''
    #print ccel_hist.values()
    #cancer_genes = cancerGenes2(sys.argv[7], ccel_type.values(), ccel_hist.values(), ccel_site.values(), ccel_site2.values())
    #print len(cancer_genes)
    combined_probes=[]
    combined_cnvs=[]
    oncoprobes=[]
    '''
    for i,c in enumerate(headers):
    	if (i-2)>=0:	
        	exp_list=[]
        	cancer_list=[]
        	mutation_list=[]
        	tsg_list=[]
        	control_list=[]
        	ess_list=[]
        	driver_list=[]
        	tf_list=[]
		for f in hist2:
			if f in genes:
                      		top_list.append(float(expr1[str(c)+str(f)]))
			if f in cnv_g:
                      		top_list.append(float(cnv[str(c)+str(f)]))
		for cc in top_compl:
                      	for p in compl[int(cc)]:
                          	if (str(c)+str(p)) in expr:
                             		exp_list.append(float(expr[str(c)+str(p)]))
                          	if (str(c)+str(p)) in cnv:
                             		exp_list.append(float(cnv[str(c)+str(p)]))
               	features['top_compl'].append(exp_list)	
		for ex in genes:

			#if p2g[ex] in cancer_genes:
			#    exp_list.append(float(expr[str(c)+str(p2g[ex])]))	
			if p2g[ex] in cancer_genes:
                      		cancer_list.append(float(expr[str(c)+str(p2g[ex])]))
                      		#combined_probes.append(ex)

			if p2g[ex] in tfs:
                      		tf_list.append(float(expr[str(c)+str(p2g[ex])]))
                      		#combined_probes.append(ex)

			if p2g[ex] in mutations:
                      		mutation_list.append(float(expr[str(c)+str(p2g[ex])]))
                      		#combined_probes.append(ex)
			if p2g[ex] in tsg:
                      		tsg_list.append(float(expr[str(c)+str(p2g[ex])]))
                      		#combined_probes.append(ex)
			if p2g[ex] in essG:
                      		ess_list.append(float(expr[str(c)+str(p2g[ex])]))
                      		#combined_probes.append(ex)
			if p2g[ex] in driverGenes:
                      		driver_list.append(float(expr[str(c)+str(p2g[ex])]))
                      		#combined_probes.append(ex)
			

		for ex in cnv_g:
			if ex in cancer_genes:	
                      		cancer_list.append(float(cnv[str(c)+str(ex)]))	
                      		#combined_cnvs.append(ex)

			if ex in tfs:	
                      		tf_list.append(float(cnv[str(c)+str(ex)]))	
                      		#combined_cnvs.append(ex)

			if ex in mutations:	
                      		mutation_list.append(float(cnv[str(c)+str(ex)]))
                      		#combined_cnvs.append(ex)
			if ex in tsg:	
                      		tsg_list.append(float(cnv[str(c)+str(ex)]))	
                      		#combined_cnvs.append(ex)
			if ex in essG:	
                      		ess_list.append(float(cnv[str(c)+str(ex)]))	
                      		#combined_cnvs.append(ex)
			if ex in driverGenes:	
                      		driver_list.append(float(cnv[str(c)+str(ex)]))	
                      		#combined_cnvs.append(ex)	

               	#features['oncogenes'].append(cancer_list+c_features[c])	
               	#features['mutations'].append(mutation_list)	
               	#features['tsg'].append(tsg_list)	
               	#features['essential'].append(ess_list)
               	#features['driver'].append(driver_list)
               	#features['tf'].append(tf_list)

    redundants=[]
    print "redundant"
    for i1,f1 in enumerate(features['oncogenes'][0]):
	row=[]
    	print i1
	for i2,f2 in enumerate(features['oncogenes'][0]):
		if i1>i2:
			row.append(pearsonr(np.array(features['oncogenes'])[:,i1],np.array(features['oncogenes'])[:,i2])[0])
		else:
			row.append(0)
	redundants.append(row)
    features['oncogenes2']=[]
    noRed=[]
    print "removing..."
    for i in range(len(redundants),0):
	if (min(redundants[i])<0.8):
		noRed.append(i)
    tmp=np.array()

    features['oncogenes2']=np.array(features['oncogenes'])[:,noRed]
    features['oncogenes2'].tolist()
    print len(features['oncogenes'][0])
    print len(features['oncogenes2'][0])
    #for i,f in enumerate(features['oncogenes'][0]):
	#if i in noRed:
		#np.concatenate(tmp,features['oncogenes
    print len(features['oncogenes'][0])
    cls, means, steps = kmeans(np.array(features['oncogenes']).T, k=3000, plus=True)
    #print len(cls)
    #print len(means)
    #print len(means[0])
    features['oncogenes']=means.T	
    features['oncogenes']=features['oncogenes'].tolist()	
    print len(features['oncogenes'][0])

    #for i,c in enumerate(headers):
    #	if (i-2)>=0:	
	#	features['oncogenes'][i-2]=features['oncogenes'][i-2]+c_features[c]
    '''
    print "features test"
    '''
    combined_probes_test=[]
    combined_cnvs_test=[]
    for i,c in enumerate(headers_t):
    	if (i-2)>=0:	
        	exp_list=[]
        	cancer_list=[]
        	mutation_list=[]
        	tsg_list=[]
        	control_list=[]
        	ess_list=[]
        	driver_list=[]
		for ex in genes:

			#if p2g[ex] in cancer_genes:
			#    exp_list.append(float(expr[str(c)+str(p2g[ex])]))	
			if p2g[ex] in cancer_genes:
                      		cancer_list.append(float(expr2b[str(c)+str(p2g[ex])]))
                      		combined_probes.append(ex)

			if p2g[ex] in mutations:
                      		mutation_list.append(float(expr2b[str(c)+str(p2g[ex])]))
                      		combined_probes_test.append(ex)
			if p2g[ex] in tsg:
                      		tsg_list.append(float(expr2b[str(c)+str(p2g[ex])]))
                      		combined_probes_test.append(ex)
			if p2g[ex] in essG:
                      		ess_list.append(float(expr2b[str(c)+str(p2g[ex])]))
                      		combined_probes_test.append(ex)
			if p2g[ex] in driverGenes:
                      		driver_list.append(float(expr2b[str(c)+str(p2g[ex])]))
                      		combined_probes_test.append(ex)

		for ex in cnv_g:
			if ex in cancer_genes:	
                      		cancer_list.append(float(cnv2[str(c)+str(ex)]))	
                      		combined_cnvs_test.append(ex)

			if ex in mutations:	
                      		mutation_list.append(float(cnv2[str(c)+str(ex)]))
                      		combined_cnvs_test.append(ex)
			if ex in tsg:	
                      		tsg_list.append(float(cnv2[str(c)+str(ex)]))	
                      		combined_cnvs_test.append(ex)
			if ex in essG:	
                      		ess_list.append(float(cnv2[str(c)+str(ex)]))	
                      		combined_cnvs_test.append(ex)
			if ex in driverGenes:	
                      		driver_list.append(float(cnv2[str(c)+str(ex)]))	
                      		combined_cnvs_test.append(ex)	

               	features_test['oncogenes'].append(cancer_list)	
               	features_test['mutations'].append(mutation_list)	
               	features_test['tsg'].append(tsg_list)	
               	features_test['essential'].append(ess_list)
               	features_test['driver'].append(driver_list)
    '''
    print "f test"
    '''
    for i,c in enumerate(headers):
    	if (i-2)>=0:	
        	exp_list=[]
		for ex in genes:

			#if p2g[ex] in cancer_genes:
			#    exp_list.append(float(expr[str(c)+str(p2g[ex])]))	
			if p2g[ex] in mutations:
                      		exp_list.append(float(expr[str(c)+str(p2g[ex])]))
                      		combined_probes.append(ex)
		for ex in cnv_g:
			if ex in mutations:	
                      		exp_list.append(float(cnv[str(c)+str(ex)]))
                      		combined_cnvs.append(ex)	
               	features['mutations'].append(exp_list)

    for i,c in enumerate(headers):
    	if (i-2)>=0:	
        	exp_list=[]
		for ex in genes:

			#if p2g[ex] in cancer_genes:
			#    exp_list.append(float(expr[str(c)+str(p2g[ex])]))	
			if p2g[ex] in tsg:
                      		exp_list.append(float(expr[str(c)+str(p2g[ex])]))
                      		combined_probes.append(ex)
		for ex in cnv_g:
			if ex in tsg:	
                      		exp_list.append(float(cnv[str(c)+str(ex)]))	
                      		combined_cnvs.append(ex)
               	features['tsg'].append(exp_list)

    for i,c in enumerate(headers):
    	if (i-2)>=0:	
        	exp_list=[]
		for ex in genes:

			#if p2g[ex] in cancer_genes:
			#    exp_list.append(float(expr[str(c)+str(p2g[ex])]))	
			if p2g[ex] in essG:
                      		exp_list.append(float(expr[str(c)+str(p2g[ex])]))
                      		combined_probes.append(ex)
		for ex in cnv_g:
			if ex in essG:	
                      		exp_list.append(float(cnv[str(c)+str(ex)]))	
                      		combined_cnvs.append(ex)
               	features['essential'].append(exp_list)
    #combined features
    combined_probes=list(set(combined_probes))
    combined_cnvs=list(set(combined_cnvs))
    features['combined']=[]
    for i,c in enumerate(headers):
    	if (i-2)>=0:	
        	exp_list=[]
		for ex in combined_probes:
                	exp_list.append(float(expr1[str(c)+str(ex)]))
		for ex in combined_cnvs:
                	exp_list.append(float(cnv[str(c)+str(ex)]))
   		features['combined'].append(exp_list)

    combined_probes_test=list(set(combined_probes_test))
    combined_cnvs_test=list(set(combined_cnvs_test))
    features_test['combined']=[]
    for i,c in enumerate(headers_t):
    	if (i-2)>=0:	
        	exp_list=[]
		for ex in combined_probes:
                	exp_list.append(float(expr2[str(c)+str(ex)]))
		for ex in combined_cnvs:
                	exp_list.append(float(cnv2[str(c)+str(ex)]))
   		features_test['combined'].append(exp_list)
  

    '''
    #print gct header
    scores={}
    for f in features:
	scores[f]=0	
    scoresp={}
    for f in features:
	scoresp[f]=0	
    print "features ready"
	
    out_file_f = open("prediction"+str(time.time())+".gct","w")
    out_file_f.write("#1.2\n")
    out_file_f.write(str(len(geneInEssent))+"\t"+str(len(headers_t)-2)+"\n")
    for c in headers_t:
        if c!="Name":
           out_file_f.write("\t")
        out_file_f.write(str(c))
    out_file_f.write("\n")
    best=0
    control=0
    avg=0
    out_file_f1 = open("0.5"+str(time.time())+".gct","w")
    out_file_f2 = open("0.3_low_std"+str(time.time())+".gct","w")
    out_file_f3 = open("negative_low_std"+str(time.time())+".gct","w")
    out_file_cluster = open("cluster"+str(time.time())+".gct","w")
    out_file_base = open("baseline"+str(time.time())+".gct","w")
    out_file_scores = open("scores"+str(time.time())+".gct","w")

    for c in leaves:
	out_file_cluster.write("\t"+str(c))
    out_file_cluster.write("\n")
    rndGenes=random.sample(geneInEssent, 1000)
    avg_cluster={}
    avg_total={}
    for cls in clusters:
    	avg_cluster[cls]=0
    	avg_total[cls]=0    
    avg_baseline=0    
    avg_combined=0    
    avg_pred_tot=0    
    avg_pred_cls=0    
    training_ccel=range(len(headers)-2)
    test_ccel=range(len(headers)-2,len(headers_t)+len(headers)-4)
    '''
    rs = cross_validation.ShuffleSplit(len(training_ccel), n_iter=1,test_size=0.25, random_state=0)	
    for train_index, test_index in rs:
	training_ccel=train_index.tolist()
	test_ccel=test_index.tolist()
    '''
    test_ccel.sort()
    #for g in rndGenes:
    for g in geneInEssent:
    #for g in notNoise:
    #for g in priority:
	
	intr+=1
	yt=np.array(essent2[g])
	#avg=np.mean(p2)
	#sd=np.std(p2)
	#for i,e in enumerate(yt)
	#	yt
	current_score={}
	current_scorep={}
	fold_score=[]
	xt=np.array(full3)
	#xt=xt[training]
	#test=np.array(full2)[training]
	#kf = KFold(len(essent2[g]), n_folds=10)

	sp2=0
	sp4=0
	s1_prediction[g]={}
	prediction=[]


	#cluster-based prediction
	pred_c={}
	pred_c_fs={}
	pred_tot={}
	whole_pred_cls=[0]*len(test_ccel)
	whole_pred_cls_fs=[]
	whole_pred_tot=[]
	whole_expected=[]
	pad=0
	#print training_ccel
	#print test_ccel
	for cls in clusters:
		pred_c[cls]=[]
		pred_c_fs[cls]=[]
		pred_tot[cls]=[]
		expected=[]
		cluster_ccel=clusters[cls]
		cluster_training=[]
		cluster_test=[]
		#new_train=leaves #debug
		for i,c in enumerate(cluster_ccel):
			if c in training_ccel:
				cluster_training.append(c)
			elif c in test_ccel:
				cluster_test.append(c)
		#print str(len(cluster_training))+"\t"+str(len(cluster_test))
		cluster_training=np.array(cluster_training)
		cluster_test=np.array(cluster_test)
		

		#eps2=3*np.std(yt[np.array(cluster_training)]) * math.sqrt( math.log(len(yt[np.array(cluster_training)])) / len(yt[np.array(cluster_training)]) )
		#eps2=eps2*0.1
		#cc=max(abs(np.mean(yt[np.array(cluster_training)]) + np.std(yt[np.array(cluster_training)])), abs(np.mean(yt[np.array(cluster_training)]) - np.std(yt[np.array(cluster_training)])))

	    	svm2 = svm.SVR(C=0.01,epsilon=0.01).fit(xt[cluster_training],yt[cluster_training])
	    	p2=svm2.predict(xt[cluster_test])
		p2=p2*-1
		#print spearmanr(np.array(p2),yt[cluster_test])[0]
		#avg_cluster[cls]+=spearmanr(np.array(p2),yt[cluster_test])[0]
		#pred_c[cls].append(p2[0]*-1)	#prediction of cluster-based model is negatively correlated with expected essentialities
		#tmp=np.array(pred_c[cls])
		#avg=np.mean(np.concatenate((p2,yt[cluster_training])))
		#sd=np.std(np.concatenate((p2,yt[cluster_training])))
		avg=np.mean(p2)
		sd=np.std(p2)
		#positions=stats.rankdata(cluster_test)
		#print cluster_test
		'''
		if cls==6:
			print p2
			print eps2
			print cc
			print yt[cluster_training]
		'''
		for i,c in enumerate(cluster_test):
			#print str(p2[i])+"\t"+str(c)
			if sd!=0:
				#print (p2[i]-avg)/sd,
				whole_pred_cls[test_ccel.index(c)]=(p2[i]-avg)/sd
				if math.isnan(whole_pred_cls[test_ccel.index(c)]):
					print cls
					print i
					print c
					print sd
					print avg
					print p2[i]
					print ""
				#whole_pred_cls[test_ccel.index(c)]=p2[i]
			else:
				whole_pred_cls[test_ccel.index(c)]=p2[i]
				
		#print whole_pred_cls
	'''
	avg_baseline+=spearmanr(np.array(whole_pred_cls),yt[test_ccel])[0]
		
	print "score:"
	print spearmanr(np.array(whole_pred_cls),yt[test_ccel])[0]
	print avg_baseline/intr
	print "---------cls-------"
	for cls in clusters:
		print avg_cluster[cls]/intr
	print "-------------------"
	print ""

	#print whole_pred_tot
	whole_pred_cls=[]
	for cls in clusters:
		whole_pred_cls=whole_pred_cls+pred_c[cls]
		whole_pred_cls_fs=whole_pred_cls_fs+pred_c_fs[cls]

	xt=np.array(full3)
	test=np.array(full3_test)
	selector = SelectPercentile(f_regression, percentile=18).fit(xt, yt)
	xt2=selector.transform(xt)
	test=selector.transform(test)

	eps2=3*np.std(yt) * math.sqrt( math.log(len(yt)) / len(yt) )
	cc=max(abs(np.mean(yt) + np.std(yt)), abs(np.mean(yt) - np.std(yt)))
	knn2 = svm.SVR(C=cc,epsilon=eps2)
	res=knn2.fit(xt2,yt).predict(test)
	'''
	out_file_f.write(g+"\t"+g)
	for p in whole_pred_cls:
		out_file_f.write("\t"+str(p))
	out_file_f.write("\n")
	out_file_f.flush()


	#best+=max(current_score)	

    out_file_f.close()
    return 0


def main(argv):
    return gep(argv)


if __name__ == "__main__":
   main(sys.argv[1:])

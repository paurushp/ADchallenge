import csv
with open('eggs.csv', 'rb') as csvfile:
from numpy import genfromtxt
my_data = genfromtxt('my_file.csv', delimiter=',')
numpy.recfromcsv(fname, delimiter=',', filling_values=numpy.nan, case_sensitive=True, deletechars='', replace_space=' ')
import csv
import numpy
reader=csv.reader(open("test.csv","rb"),delimiter=',')
x=list(reader)
result=numpy.array(x).astype('float')
numpy.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1)

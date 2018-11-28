"""
Author: Xu Ma.
Data:10/31/2018
Email:xuma@my.unt.edu
Data source: http://dcslab.cse.unt.edu/~song/Data/
"""
import numpy as np

data_dir='/home/xuma/data/Anomaly/datasets/bak/'
filename='20110920.txt'
save_data_dir='/home/xuma/data/Anomaly/datasets/'

#read the first line of file
f=open(data_dir+filename)
print(f.readlines()[0])


data=np.genfromtxt(data_dir+filename,autostrip=True,delimiter='\t')
print(data[:,1])
# data=np.loadtxt(data_dir+filename)
np.savetxt(save_data_dir+filename,data)
print(data.shape)



# newString=re.sub('"34fgd5765"','',testString)
# print(newString)






# with open(data_dir+filename) as file_object:
#     contents=file_object.read()
#     print(contents)







#data=np.loadtxt(data_dir+filename,dtype=np.str)








# with open(data_dir+filename) as f:
#    for line in f:
#        print(line)
#        data=np.fromstring(line., dtype=str,sep='\t')
















# data=np.loadtxt(data_dir+filename,dtype=np.str)
# print(data.shape)

# with open(data_dir+filename) as f:
#     first_line = f.readline()
#
# print(first_line)

# f=open(data_dir+filename,'r')
# lines=f.readlines()
# # print(lines[0])
# print(lines[100])
# for line in lines:
#     print(line)
#     line_value=np.fromstring(line, dtype=np.str)
#     print(line_value.shape)
#



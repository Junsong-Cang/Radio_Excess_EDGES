import os
import random
import shutil

fR = 2.8E0
L_X = 39.76

Len=30

code=str(random.randint(0,9))
for id in range(0,Len):
	a=random.randint(0,9)
	code = code + str(a)

H5File = 'home/dm/watson/21cmFAST-data/LC_'+code+'.h5'
ParamFile = 'inputs/MCMC_input_'+code+'.ini'
cmd = 'python3 Run_21cmFAST.py < ' + ParamFile

print(H5File)
print(ParamFile)
print(cmd)

shutil.copyfile('inputs/input_template.ini',ParamFile)

f=open(ParamFile,'a')
f.write('fR = ')
f.write(str(fR))
f.write('\n')
f.write('L_X = ')
f.write(str(L_X))
f.write('\n')
f.write("FileName = '")
f.write(H5File)
f.write("'")
f.write('\n')
f.write('\n')
f.close

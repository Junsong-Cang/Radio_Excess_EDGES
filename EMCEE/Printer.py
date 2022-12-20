import os
import random

Len=30

code=str(random.randint(0,9))

for id in range(0,Len):
	a=random.randint(0,9)
	code = code + str(a)

H5File = 'LC'+code+'.h5'
ParamFile = 'input_'+code+'.ini'
cmd = 'python3 main.py < ' + ParamFile

print(H5File)
print(ParamFile)
print(cmd)


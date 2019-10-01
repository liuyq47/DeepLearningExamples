import numpy as np
import sys

count = sys.argv[1]
hosts = np.random.choice(np.arange(1,34), size=32, replace=False)

f = open('host'+count,'w')
for i in hosts:
	f.write('h'+str(i)+'\n')

f.close()


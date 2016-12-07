import sys
import os

if len(sys.argv) < 2:
	print 'Usage correctPath.py filetoCorrect'
	sys.exit(1)

with open(sys.argv[1], "w") as f:
	f.seek(0,os.SEEK_END)
	size = f.tell()
	f.seek(0,os.SEEK_SET)
	for line in f:
		sys.stdout.write(f.tell()/size)
		sys.stdout.write('/'+str(size))
		s = line.split(' ')
		
		

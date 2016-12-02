import glob
import os.path as path
import sys
from sets import Set

if len(sys.argv) < 3:
	print("Usage : python createListImage.py imagedir savedFile")
	sys.exit(1)

def browsedir(dir):
	if '*' in dir:
		return Set()
	print('Explor :', dir)
	imList = Set()
	for file in glob.glob(path.join(dir, '*')):
		if path.isdir(file):
			imList |= browsedir(file)
		elif '.jpg' in file or '.JPG' in file or '.JPEG' in file:
			imList.add(file)	
	return imList

l = browsedir(sys.argv[1])
with open(sys.argv[2], 'w') as f:
	for im in l:
		f.write(im)
		f.write('\n')


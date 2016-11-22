import numpy as np
import lmdb
import glob
import sys
import os
from PIL import Image

size = 128

if len(sys.argv) < 3:
	print('Usage : python createLMDB imagenet <path to imagenet> <path to save dir>')
	sys.exit(1)

trainpath = os.path.join(sys.argv[1], 'train')
valpath = os.path.join(sys.argv[1], 'val')

savedtrain = os.path.join(sys.argv[2], 'train')
savedval = os.path.join(sys.argv[2], 'val')

if not os.path.isdir(savedtrain):
	os.mkdir(savedtrain)
for concept in glob.glob(os.path.join(trainpath, '*')):
	print('Concept :', concept)
	if not '*' in concept:
		savedDir = os.path.join(savedtrain, os.path.basename(concept))
		print(savedDir)
		for imName in glob.glob(os.path.join(concept, '*')):
			#print("Image :", imName)
			im = Image.open(imName).convert("RGB")
			im = im.resize((size,size))
			
			if not os.path.isdir(savedDir):
				os.mkdir(savedDir)
			#print("\tSave :", os.path.join(savedDir,os.path.basename(imName)))
			im.save(os.path.join(savedDir, os.path.basename(imName)))

#if not os.path.isdir(savedval):
#	os.mkdir(savedval)
#for concept in glob.glob(os.path.join(valpath,'*')):
#	print("Concept :", concept)
#	if not '*' in concept:
#		for imName in glob.glob(os.path.join(concept, '*')):
#			im = Image.open(imName).convert("RGB")
#			im = im.resize((size,size))
#
#			savedDir = os.path.join(savedval, os.path.basename(concept))
#
#			if not os.path.isdir(savedDir):
#				os.mkdir(savedDir)
#			im.save(os.path.join(savedDir, os.path.basename(imName)))

#-*- coding: UTF-8 -*-
# coding=gbk

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import json 
import sys
import cv2
import numpy
import numpy as np
import time
import csv

from pylab import plot, show, title, xlabel, ylabel, subplot, savefig
from numpy import sin, linspace, pi
import os
from os.path import isfile, join
from os import listdir
#infile =open('imdbids_boxer.txt','t')



start_time = time.clock()

caffe_root = '../../caffe/'
import sys
sys.path.insert(0,caffe_root + 'python')

import caffe


class bcolors:
    HEADER = '\033[95m'#purple
    OKBLUE = '\033[94m'#blue
    OKGREEN = '\033[92m'#green
    WARNING = '\033[93m'#yello
    FAIL = '\033[91m'#red
    ENDC = '\033[0m'

def meancolor(img):
	h= img.shape[0]
	w= img.shape[1]

	mean0=0
	mean1=0
	mean2=0
	for i in range(0,h):
		for j in range(0,w):
			mean0=mean0+img[i,j][0]
			mean1=mean1+img[i,j][1]
			mean2=mean2+img[i,j][2]

	for i in range(0,h):
		for j in range(0,w):
			img[i,j][0]=mean0/h/w
			img[i,j][1]=mean1/h/w
			img[i,j][2]=mean2/h/w
	pvalues=[mean0/h/w,mean1/h/w,mean2/h/w]
	return img,pvalues

def meancolor2(img):
	h= img.shape[0]
	w= img.shape[1]
	mean0=0
	mean1=0
	mean2=0
	for i in range(0,h):
		for j in range(0,w):
			mean0=mean0+img[i,j][0]
			mean1=mean1+img[i,j][1]
			mean2=mean2+img[i,j][2]	
	pvalues=[mean0/h/w,mean1/h/w,mean2/h/w]
	return pvalues
### get the dominating color of a pixel
def coloranalize(p):#Get the dominating elements from [B,G,R] 
	
	#print p,
	if max(p)-min(p)<10:
		return 'GREY',0
	elif (p[0]>p[1])and(p[0]>p[2]):		
		return 'B',p[0]-min(p[1],p[2])
	elif (p[1]>p[0])and(p[1]>p[2]):
		return 'G',p[1]-min(p[0],p[2])
	elif  (p[2]>p[0])and(p[2]>p[1]):
		return 'R',p[2]-min(p[0],p[1])
	else:
		return 'N',0

def imgcut(img,n):
	meanimg=img
		
	h= img.shape[0]
	w= img.shape[1]
	#print h,w
	pvector=[]
	for i in range(0,n):
		for j in range(0,n):
			im,p=meancolor(img[0+h*i/n:h*i/n+h/n, 0+w*j/n:w*j/n+w/n])
			meanimg[0+h*i/n:h*i/n+h/n, 0+w*j/n:w*j/n+w/n]=im
			pvector.append(p)
			
	return meanimg,pvector



################################################################





scenelog = open("../scene_log/casino_royale.txt", "w")
# MODEL_FILE = '../../googlenet_places205/deploy_places205.prototxt'
# PRETRAINED = '../../googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel'
MODEL_FILE = '/home/vionlabs/Documents/scene_classification/Places_CNDS_model/deploy.prototxt'
PRETRAINED = '/home/vionlabs/Documents/scene_classification/Places_CNDS_model/8conv3fc_DSN.caffemodel'
# MODEL_FILE = '../../placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt'
# PRETRAINED = '../../placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel'

#VIDEOPATH = '/home/vionlabs/Documents/weilun_thesis/caffe/examples/test_data/Life_of_Pi.avi'
VIDEOPATH = '/home/vionlabs/Documents/scene_classification/video_file/casino_royale.mp4'
# mean = np.load('../Places_CNDS_model/places_mean.npy')
# print mean.shape


#caffe.set_phase_test()
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load('../Places_CNDS_model/places_mean.npy').mean(1).mean(1),
                       channel_swap=(0,1,2),
                       raw_scale=255,
                       image_dims=(256, 256))



all=[]
dir='/mnt/movies03/boxer_movies/'

num=0
num1=0
num2=0
vfiles={}
for line in listdir(dir) :
	#print line 
	subdir=dir+line+'/'
	if len(listdir(subdir)) == 2:
		num+=1
		for f in listdir(subdir):
			
			if f=='done.txt':				
				continue
			if not os.path.isdir(subdir+f+'/'):
				#print f
				num1+=1
				vfiles[line]=subdir+f
			else:
				#print '######'
				sign=0
				for ff in listdir(subdir+f):
					#print ff
					if ff.endswith('.avi') or ff.endswith('.mkv') or ff.endswith('.mp4') or ff.endswith('.avi'):
						fff=ff
						sign+=1
						continue
				if sign==1:
					num2+=1
					vfiles[line]=subdir+f+'/'+fff
					#print subdir+f+'/'+fff
			# else:

			# 	print listdir(subdir+f)

#print vfiles
movienum=0
for f in vfiles:
	movienum+=1
	dir='fingerpringting_info/'+f
	print dir
	if not os.path.exists(dir): 
		os.makedirs(dir)
	else:
		continue



	### starting point
	start=time.time()
	#allinfo=open('','')
	of=open(dir+"/basicinfo.json","w")
	print vfiles[f]
	capture = cv2.VideoCapture(vfiles[f])

	### pixel dif calculation
	ppvector=difpvector= [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

	print ppvector
	framenum=0
	change=[]
	allinfo=[] ### restore allinfo like '212 89 11 2979'
	avecolor=[]
	dcolor=[]
	dvalue=[]
	dark=[]
	while(capture.isOpened()):
		
		#if framenum>2000:
		#	break
		ret, frame = capture.read()
		# print frame_riginal.shape
		# frame = cv2.resize(frame_riginal, (0,0), fx=float(400)/float(frame_riginal.shape[1]), fy=float(400)/float(frame_riginal.shape[1]))
		#print frame.shape
		if frame is None:
			break
		framenum=framenum+1
		

		if framenum % 500 == 160:
			open_cv_frame = np.asarray(frame).astype(np.float32)
			movie_frame = np.divide(open_cv_frame,255)
			#input_image = caffe.io.load_image(movie_frame)
			#cv2.imshow('frame', movie_frame)
			print '--------------------'
			print movie_frame.shape
			prediction = net.predict([movie_frame])
			print max(prediction[0])
			result = prediction[0]
			m = max(result)
			index = result.argmax(axis=0)
			print 'Category number:', index
			#with open('../../placesCNN/category_filter.csv', 'rb') as f:
			with open('categoryIndex_places205.csv', 'rb') as f:
				reader = csv.reader(f)
				for i, row in enumerate(reader):
					if i == index:
					    print row[0]
					    scenelog.write(str(index) + os.linesep)
					    scenelog.write(str(row[0]) + os.linesep)
					    scenelog.write(str(m) + os.linesep)
					i += 1
			print time.clock() - start_time, "seconds"

			print "#######"
			print bcolors.OKBLUE+str(movienum)+' '+str(len(vfiles))+bcolors.ENDC
			print framenum
			print str((end-start)/60)+' M'

		small = cv2.resize(frame, (0,0), fx=0.05, fy=0.05)
		#cv2.imshow("Original", cv2.resize(frame, (0,0), fx=0.4, fy=0.4))
		#cv2.waitKey(0)
		mcolor=meancolor2(small)
		if mcolor[0]+mcolor[1]+mcolor[2]<60:
			dark.append(1)
		else:
			dark.append(0)
		avecolor.append(mcolor)
		domicolor,domivalue=coloranalize(mcolor)
		dcolor.append(domicolor)
		dvalue.append(domivalue)
		img0,pvector=imgcut(small,10)
		
		#print pvector
		#print ppvector
		for x in range(100):
			difpvector[x][0]=int(pvector[x][0]-ppvector[x][0])
			difpvector[x][1]=int(pvector[x][1]-ppvector[x][1])
			difpvector[x][2]=int(pvector[x][2]-ppvector[x][2])
		
		meanabsdif=[]

		up10=0
		up50=0
		up=0
		down=0
		for v in difpvector:
			m=v[0]+v[1]+v[2]
			if m>0:
				up=up+1
			if m<0:
				down=down+1
			absm=abs(m)/3
			if absm>10:
				up10=up10+1
			if absm>50:
				up50=up50+1
			meanabsdif.append(absm)
		# if up>90:
		# 	print bcolors.WARNING+'↑'+bcolors.ENDC
		# if down>90:
		# 	print bcolors.WARNING+'↓'+bcolors.ENDC
		
		#print meanabsdif
		change.append(up10)
		
		addition=0
		for e in meanabsdif:
			addition=addition+e
		
		of.write(str(framenum)+' '+str(up10)+' '+str(up50)+' '+str(addition)+' '+str(mcolor[0])+' '+str(mcolor[1])+' '+str(mcolor[2])+'\n')
		allinfo.append(str(framenum)+' '+str(up10)+' '+str(up50)+' '+str(addition))
		#print len (difpvector)
		ppvector=pvector
		#cv2.imshow("mean", img0)
		#cv2.waitKey(0)

		end=time.time()

	#x=linspace(0, framenum, len(change))
	#y=change
	#plot (x,y)
	#show()



	### shot cut detection 
	print allinfo 
	filtered=[]
	res=[]
	sign=0
	change=[]
	for line in allinfo:
		
		if sign is 0:
			lold=line.replace('\n','').split(' ')
			sign=sign+1
			continue
		if sign is 1:	
			l=line.replace('\n','').split(' ')
			sign=sign+1
			continue
		if sign >= 2:
			lnew=line.replace('\n','').split(' ')
			sign=sign+1
			#continue
		
		if int(l[3])/1.5>(int(lold[3])+int(lnew[3])):
			
			print lold[3],l[3],lnew[3]
			if int(l[1])>=5:
				change.append((int(lold[3])+int(lnew[3]))/2)
				filtered.append(l[0])
				res.append(l[0])
				#if l[0] in gt:
				#	sum.append(l[0])
			else:
				change.append(l[3])

		else:	
			change.append(l[3])
		lold=l
		l=lnew
		#lnew
		#break
		print '##########'

	print filtered
	height=[]
	count=0
	add=0
	mean=0
	sign=0
	lnum=0
	for line in allinfo:
		l=line.replace('\n','').split(' ')
		lnum=l[0]
		#break
		if l[0] in filtered:
			#print '##########'
			print l[0]
			mean=add/count
			while sign<(int(l[0])-1):
				sign=sign+1
				height.append(mean)
			count=0
			add=0
			print '##########'
		else:
			#print l[3]
			count=count+1
			#print l[3]
			add=add+int(l[3])
			#print add 


	mean=add/count
	while int(sign)<int(lnum):
		sign=sign+1
		height.append(mean)



	print len(height)
	print max(height)*2, len(avecolor)
	of1=open(dir+'/shotchange.txt','w')
	for line in filtered:
		of1.write(str(line)+'\n')



	infile1=open(dir+'/shotchange.txt','r')
	of=open(dir+'/colorinfo.txt','w')
	shotchange=[]
	for line in infile1:
		shotchange.append(line.replace('\n','')) 

	infile2=open(dir+'/basicinfo.json','r')
	#dataall={}
	#dataall['Imdbid']='tt0468569'
	#dataall['ColorInfo']=[]
	for line in infile2:
		data={}
		l= line.replace('\n','').split(' ')
		print l 
		data['FrameNumber']=l[0]
		data['Change10']=l[1]
		data['Change50']=l[2]
		data['Difference']=l[3]
		data['AveragePixelValue']={}
		data['AveragePixelValue']['B']=l[4]
		data['AveragePixelValue']['G']=l[5]
		data['AveragePixelValue']['R']=l[6]
		if str(l[0]) in shotchange:
			data['IfShotChangeHere']=1
		else:
			data['IfShotChangeHere']=0
		print data
		#dataall['ColorInfo'].append(data)
		#if l[0] ==str(3):
		#	break
		json.dump(data,of)
		of.write('\n')


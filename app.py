#flask requirements
from flask import Flask,render_template,request,redirect
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

#OCR code requirements
import cv2
import pytesseract
import numpy as np
import json
pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route("/uploader",methods=['GET','POST'])
def uploader():
    if request.method=="POST":
        f = request.files['file1']
        n = request.form['person']
        print(n)
        p=f.filename
        x=p.split(".")
        f.filename=n+'.'+x[1]
        ff=os.path.join("C:\\Users\\AMAN\Desktop\\OCR\\static\\user_form",secure_filename(f.filename))
        f.save(ff)
        image_path='C:\\Users\\AMAN\\Desktop\\OCR\\static\\user_form\\'+(f.filename)

        per=25
        roi=[[(36, 142), (162, 168), 'text', 'first name'],
             [(310, 146), (450, 166), 'text', 'last name'],
             [(38, 234), (92, 258), 'text', 'month'],
             [(102, 240), (160, 254), 'text', 'day'], 
             [(180, 238), (226, 260), 'text', 'year'], 
             [(252, 240), (452, 262), 'text', 'gender'],
             [(34, 330), (228, 352), 'text', 'e-mail'],
             [(254, 330), (448, 352), 'text', 'mobile no'], 
             [(34, 426), (230, 444), 'text', 'phone no'], 
             [(252, 424), (446, 446), 'text', 'work no'],
             [(36, 502), (232, 524), 'text', 'company'],
             [(38, 576), (228, 602), 'text', 'courses']]

             
        imgF=cv2.imread('C:\\Users\\AMAN\\Desktop\\OCR\\static\\images\\regform.jpeg')
        h,w,c=imgF.shape
        imgF=cv2.resize(imgF,(w,h))
        orb = cv2.ORB_create(5000)
        kp1,des1=orb.detectAndCompute(imgF,None)
        img = cv2.imread(image_path)
        h,w,c=img.shape
        img=cv2.resize(img,(w,h))
        #cv2.imshow('frame1',img)
        kp2,des2=orb.detectAndCompute(img,None)
        bf=cv2.BFMatcher(cv2.NORM_HAMMING)
        matches=bf.match(des2,des1)
        matches.sort(key=lambda x:x.distance)
        good=matches[:int(len(matches)*(per/100))]
        imgMatch = cv2.drawMatches(img,kp2,imgF,kp1,good,None,flags=2)
        imgMatch=cv2.resize(imgMatch,(w,h))
        #cv2.imshow('frame1',imgMatch)
        srcpoints=np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstpoints=np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M,_=cv2.findHomography(srcpoints,dstpoints,cv2.RANSAC,5.0)
        imgscan=cv2.warpPerspective(img,M,(w,h))
        imgscan=cv2.resize(imgscan,(w,h))
        #cv2.imshow('frame1',imgscan)
        # cv2.imshow("keypoints",imgkp1)
        #cv2.imshow('kanishka',imgF)
        imgshow=imgscan.copy()
        imgMask=np.zeros_like(imgshow)
        myData=[]
        for x,r in enumerate(roi):
            cv2.rectangle(imgMask,(r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
            imgshow=cv2.addWeighted(imgshow,0.99,imgMask,0.1,0)
            imgCrop=imgscan[r[0][1]:r[1][1],r[0][0]:r[1][0]]
            cv2.imshow(str(x),imgCrop)
            if r[2]=='text':
                myData.append(pytesseract.image_to_string(imgCrop))
            cv2.putText(imgshow,str(myData[x]),(r[0][0],r[0][1]),cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)
        #imgshow=cv2.resize(imgshow,(w,h))
        cv2.imshow('frame1',imgshow)
        
        with open('DataOutput.json','a+') as f:
            json.dump(myData,f)
        
    return render_template('home.html')
        
if __name__ == '__main__':
   app.run(debug = True)
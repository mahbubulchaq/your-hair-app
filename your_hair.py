import streamlit as st
from streamlit_option_menu import option_menu
from numpy import *
import numpy as np #for mathematical calculations
import cv2 #for face detection and other image operations
import dlib #for detection of facial landmarks ex:nose,jawline,ey es
from sklearn.cluster import KMeans #for clustering
from time import sleep
from PIL import Image
from itertools import cycle

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options = ["Home", "Capture", "About", "Help"],
        icons = ["house", "camera", "journal-text", "question-circle" ],
        menu_icon= "menu-app-fill",
        default_index= 0,
        orientation= "vertical"
    )

if selected == "Home":
    
    st.title(f" {selected}")
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True) 
    st.subheader("Potongan rambut terpopuler")
    filteredImages = ['shop\Comb-Over.jpg',
                    'shop\Messy-Hair.jpg',
                    'shop\Buzz-Cut-1.jpg',
                    'shop\Comma-Hair.jpg',
                    'shop\long-hair.jpg',
                    'shop\mullet2.jpg',
                    'shop\classic-middle-part.jpg',
                    'shop\pompador.jpg',
                    'shop\Mohawk.jpg'] # your images here
    caption = ['Comb Over',
            'Messy Hair',
            'Buzz Cut',
            'Comma Hair',
            'Long Hair',
            'Mulet',
            'Classic Middle Part',
            'Pompador',
            'Mohawak'] # your caption here
    cols = cycle(st.columns(4)) # st.columns here since it is out of beta at the time I'm writing this
    for idx, filteredImage in enumerate(filteredImages):
        next(cols).image(filteredImage, width=150, caption=caption[idx])

if selected == "Capture":
    st.title(f" {selected}")

    agree = st.checkbox('run camera')

    if agree:   
        picture = st.camera_input("Take a picture")

        if picture:
            with open ('test.jpg','wb') as file:
                file.write(picture.getbuffer())
                
            procces = st.button('Procces')
            if procces == True:
                #load the image
                imagepath = "test.jpg"
                #haarcascade for detecting faces
                # link = https://github.com/opencv/opencv/tree/master/data/haarcascades
                face_cascade_path = "haarcascade_frontalface_default.xml"
                #.dat file for detecting facial landmarks
                #download file path = http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
                predictor_path = "shape_predictor_68_face_landmarks.dat"
                #create the haar cascade for detecting face and smile
                faceCascade = cv2.CascadeClassifier(face_cascade_path)

                #create the landmark predictor
                predictor = dlib.shape_predictor(predictor_path)

                #read the image
                image = cv2.imread(imagepath)
                #resizing the image to 000 cols nd 500 rows
                image = cv2.resize(image, None, fx=1, fy=1, interpolation = cv2.INTER_AREA) 
                #making another copy
                original = image.copy()

                #convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                #apply a Gaussian blur with a 3 x 3 kernel to help remove high frequency noise
                gauss = cv2.GaussianBlur(gray,(3,3), 0)

                #Detect faces in the image
                faces = faceCascade.detectMultiScale(
                    gauss,
                    scaleFactor=1.05,
                    minNeighbors=5,
                    minSize=(100,100),
                    flags=cv2.CASCADE_SCALE_IMAGE
                    )
                #Detect faces in the image
                print("found {0} faces!".format(len(faces)) )

                for (x,y,w,h) in faces:
                    #draw a rectangle around the faces
                    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
                    #converting the opencv rectangle coordinates to Dlib rectangle
                    dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                    #detecting landmarks
                    detected_landmarks = predictor(image, dlib_rect).parts()
                    #converting to np matrix
                    landmarks = np.matrix([[p.x,p.y] for p in detected_landmarks])
                    #landmarks array contains indices of landmarks.
                    #"""
                    #copying the image so we can we side by side
                    #landmark = image.copy()
                    #for idx, point in enumerate(landmarks):
                            #pos = (point[0,0], point[0,1] )
                            #annotate the positions
                            #cv2.putText(landmark,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255) )
                            #draw points on the landmark positions 
                            #cv2.circle(landmark, pos, 3, color=(0,255,255))
                    
                #cv2.imshow("Landmarks by DLib", landmark)
                #"""
                #making another copy  for showing final results
                results = original.copy()

                for (x,y,w,h) in faces:
                    #draw a rectangle around the faces
                    cv2.rectangle(results, (x,y), (x+w,y+h), (0,255,0), 2)
                    #making temporary copy
                    temp = original.copy()
                    #getting area of interest from image i.e., forehead (25% of face)
                    forehead = temp[y:y+int(0.25*h), x:x+w]
                    rows,cols, bands = forehead.shape
                    X = forehead.reshape(rows*cols,bands)
                    #"""
                    #Applying kmeans clustering algorithm for forehead with 2 clusters 
                    #this clustering differentiates between hair and skin (thats why 2 clusters)
                    #"""
                    #kmeans
                    kmeans = KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10, random_state=0)
                    y_kmeans = kmeans.fit_predict(X)
                    for i in range(0,rows):
                        for j in range(0,cols):
                            if y_kmeans[i*cols+j]==True:
                                forehead[i][j]=[255,255,255]
                            if y_kmeans[i*cols+j]==False:
                                forehead[i][j]=[0,0,0]
                    #Steps to get the length of forehead
                    #1.get midpoint of the forehead
                    #2.travel left side and right side
                    #the idea here is to detect the corners of forehead which is the hair.
                    #3.Consider the point which has change in pixel value (which is hair)
                    forehead_mid = [int(cols/2), int(rows/2) ] #midpoint of forehead
                    lef=0 
                    #gets the value of forehead point
                    pixel_value = forehead[forehead_mid[1],forehead_mid[0] ]
                    for i in range(0,cols):
                        #enters if when change in pixel color is detected
                        if forehead[forehead_mid[1],forehead_mid[0]-i].all()!=pixel_value.all():
                            lef=forehead_mid[0]-i
                            break;
                    left = [lef,forehead_mid[1]]
                    rig=0
                    for i in range(0,cols):
                        #enters if when change in pixel color is detected
                        if forehead[forehead_mid[1],forehead_mid[0]+i].all()!=pixel_value.all():
                            rig = forehead_mid[0]+i
                            break;
                    right = [rig,forehead_mid[1]]
                    
                #drawing line1 on forehead with circles
                #specific landmarks are used. 
                line1 = np.subtract(right+y,left+x)[0]
                cv2.line(results, tuple(x+left), tuple(y+right), color=(0,255,0), thickness = 2)
                cv2.putText(results,' Line 1',tuple(x+left),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
                cv2.circle(results, tuple(x+left), 5, color=(255,0,0), thickness=-1)    
                cv2.circle(results, tuple(y+right), 5, color=(255,0,0), thickness=-1)        

                #drawing line 2 with circles
                linepointleft = (landmarks[1,0],landmarks[1,1])
                linepointright = (landmarks[15,0],landmarks[15,1])
                line2 = np.subtract(linepointright,linepointleft)[0]
                cv2.line(results, linepointleft,linepointright,color=(0,255,0), thickness = 2)
                cv2.putText(results,' Line 2',linepointleft,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
                cv2.circle(results, linepointleft, 5, color=(255,0,0), thickness=-1)    
                cv2.circle(results, linepointright, 5, color=(255,0,0), thickness=-1)    

                #drawing line 3 with circles
                linepointleft = (landmarks[3,0],landmarks[3,1])
                linepointright = (landmarks[13,0],landmarks[13,1])
                line3 = np.subtract(linepointright,linepointleft)[0]
                cv2.line(results, linepointleft,linepointright,color=(0,255,0), thickness = 2)
                cv2.putText(results,' Line 3',linepointleft,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
                cv2.circle(results, linepointleft, 5, color=(255,0,0), thickness=-1)    
                cv2.circle(results, linepointright, 5, color=(255,0,0), thickness=-1)    

                #drawing line 4 with circles                            
                linepointbottom = (landmarks[8,0],landmarks[8,1])
                linepointtop = (landmarks[8,0],y)
                line4 = np.subtract(linepointbottom,linepointtop)[1]
                cv2.line(results,linepointtop,linepointbottom,color=(0,255,0), thickness = 2)
                cv2.putText(results,' Line 4',linepointbottom,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0), thickness=2)
                cv2.circle(results, linepointtop, 5, color=(255,0,0), thickness=-1)    
                cv2.circle(results, linepointbottom, 5, color=(255,0,0), thickness=-1) 
                
                st.write('pajang line 1 = ',line1)
                st.write('pajang line 2 = ',line2)
                st.write('pajang line 3 = ',line3)            
                st.write('pajang line 4 = ',line4)

                similarity = np.std([line1,line2,line3])
                #print("similarity=",similarity)
                ovalsimilarity = np.std([line2,line4])
                #print('diam=',ovalsimilarity)

                #we use arcustangens for angle calculation
                ax,ay = landmarks[3,0],landmarks[3,1]
                bx,by = landmarks[4,0],landmarks[4,1]
                cx,cy = landmarks[5,0],landmarks[5,1]
                dx,dy = landmarks[6,0],landmarks[6,1]
                import math
                from math import degrees
                alpha0 = math.atan2(cy-ay,cx-ax)
                alpha1 = math.atan2(dy-by,dx-bx)
                alpha = alpha1-alpha0
                angle = abs(degrees(alpha))
                angle = 180-angle

                for i in range(1):
                    if similarity<20:
                        if angle<160:
                            st.caption('Garis rahang lebih bersudut, garis 1,2,3 dan 4 hampir sama')
                            st.markdown('Image calculation')
                            output = np.concatenate((original,results), axis=1)
                            cv2.imwrite("output.jpg", output)
                            image = Image.open("output.jpg")
                            st.image(image, caption='result image.', use_column_width=True)
                            st.markdown('Bentuk wajah anda :')
                            st.subheader('PERSEGI')
                            st.markdown('Rekomendasi potongan rambut untuk anda')
                            filteredImages = ['shop\Comb-Over.jpg',
                                            'shop\Messy-Hair.jpg'] # your images here
                            caption = ['Comb Over',
                                    'messy Hair'] # your caption here
                            cols = cycle(st.columns(4)) # st.columns here since it is out of beta at the time I'm writing this
                            for idx, filteredImage in enumerate(filteredImages):
                                next(cols).image(filteredImage, width=250, caption=caption[idx])
                            break
                        else:
                            st.markdown('Garis rahang tidak terlalu bersudut')
                            st.markdown('Image calculation')
                            output = np.concatenate((original,results), axis=1)
                            cv2.imwrite("output.jpg", output)
                            image = Image.open("output.jpg")
                            st.image(image, caption='result image.', use_column_width=True)
                            st.markdown('Bentuk wajah anda :')
                            st.subheader('BULAT')
                            filteredImages = ['shop\too.jpeg',
                                            'shop\pompador.jpg'] # your images here
                            caption = ['Two Block',
                                    'Pompador'] # your caption here
                            cols = cycle(st.columns(4)) # st.columns here since it is out of beta at the time I'm writing this
                            for idx, filteredImage in enumerate(filteredImages):
                                next(cols).image(filteredImage, width=250, caption=caption[idx])
                            break
                    if line3>line2>line1:
                        if angle<160:
                            st.caption('Dahi lebih lebar')
                            st.markdown('Image calculation')
                            output = np.concatenate((original,results), axis=1)
                            cv2.imwrite("output.jpg", output)
                            image = Image.open("output.jpg")
                            st.image(image, caption='result image.', use_column_width=True)
                            st.markdown('Bentuk wajah anda :')
                            st.subheader('SEGITIGA')
                            st.markdown('Rekomendasi potongan rambut untuk anda')
                            filteredImages = ['shop\Comb-Over.jpg',
                                            'shop\Messy-Fringe.jpg'] # your images here
                            caption = ['comb-over',
                                    'Messy Fringe',] # your caption here
                            cols = cycle(st.columns(4)) # st.columns here since it is out of beta at the time I'm writing this
                            for idx, filteredImage in enumerate(filteredImages):
                                next(cols).image(filteredImage, width=250, caption=caption[idx])
                            break
                    if ovalsimilarity<10:
                        st.caption('garis 2 dan 4 serupa dan garis 2 sedikit lebih lebar')
                        st.markdown('Image calculation')
                        output = np.concatenate((original,results), axis=1)
                        cv2.imwrite("output.jpg", output)
                        image = Image.open("output.jpg")
                        st.image(image, caption='result image.', use_column_width=True)
                        st.markdown('Bentuk wajah anda :')
                        st.subheader('DIAMOND')
                        st.markdown('Rekomendasi potongan rambut untuk anda')
                        filteredImages = ['shop\Comb-Over.jpg',
                                        'shop\slick-pompador.jpg'] # your images here
                        caption = ['Comb Over',
                                'Slick Pompador'] # your caption here
                        cols = cycle(st.columns(4)) # st.columns here since it is out of beta at the time I'm writing this
                        for idx, filteredImage in enumerate(filteredImages):
                            next(cols).image(filteredImage, width=250, caption=caption[idx])
                        break
                    if line4>line2 + 0.5:
                        if angle<160:
                            st.caption('panjang wajah terbesar dan garis rahang bersudut')
                            st.markdown('Image calculation')
                            output = np.concatenate((original,results), axis=1)
                            cv2.imwrite("output.jpg", output)
                            image = Image.open("output.jpg")
                            st.image(image, caption='result image.', use_column_width=True)
                            st.markdown('Bentuk wajah anda :')
                            st.subheader('PERSEGI PANJANG')
                            st.markdown('Rekomendasi potongan rambut untuk anda')
                            filteredImages = ['shop\cepak.jpeg',
                                            'shop\mullet2.jpg'] # your images here
                            caption = ['Cepak Haircut',
                                    'Mullet'] # your caption here
                            cols = cycle(st.columns(4)) # st.columns here since it is out of beta at the time I'm writing this
                            for idx, filteredImage in enumerate(filteredImages):
                                next(cols).image(filteredImage, width=250, caption=caption[idx])
                            break;
                        else:
                            st.caption('panjang wajah terbesar dan garis rahang tidak bersudut')
                            st.markdown('Image calculation')
                            output = np.concatenate((original,results), axis=1)
                            cv2.imwrite("output.jpg", output)
                            image = Image.open("output.jpg")
                            st.image(image, caption='result image.', use_column_width=True)
                            st.markdown('Bentuk wajah anda :')
                            st.subheader('OBLONG')

                            st.markdown('Rekomendasi potongan rambut untuk anda')
                            filteredImages = ['shop\mullet2.jpg','shop\pompador.jpg'] # your images here
                            caption = ['Mullet',
                                    'Pompador'] # your caption here
                            cols = cycle(st.columns(2)) # st.columns here since it is out of beta at the time I'm writing this
                            for idx, filteredImage in enumerate(filteredImages):
                                next(cols).image(filteredImage, width=250, caption=caption[idx])
                            break;
                    st.text("Damn! Contact the developer")

if selected == "About":
    #st.title(f" {selected}")
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    co1, co2 = st.columns(2)

    with co2:
        st.subheader('Your Hair')
        st.markdown('Membantu kamu mencari potongan rambut yang sesuai dengan bentuk wajahmu.')
        st.markdown('''Dengan perkembangan teknologi yang sangat pesat, 
        kamu dimudahkan hampir dari semua faktor, 
        salah satunya dibidang pengetahuan. 
        Disini kamu dapat mengetahui bentuk wajahmu. 
        Selain itu kamu juga dapat melihat rekomendasi 
        potongan rambut yang sesuai dengan bentuk wajahmu''')
        st.markdown('Semoga membantu :)')
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

    with col5:
        st.image('shop\ig.png', caption='llllupus', width= 44)
    with col6:
        st.image('shop\yt.png', caption='eyeeore', width= 55)

if selected == "Help":
    #st.title(f" {selected}")
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

    co1, co2 = st.columns(2)

    with co1:
        st.subheader('Cara menggunakan website Your Hair')
        st.markdown('1. Buka website your hair.')
        st.markdown('2. kemudian buka halaman capture')
        st.markdown('3. Setelah itu tekan tombol ceckbox.')
        st.markdown('4. Jika camera sudah menyala, ambil gambar wajahmu.')
        st.markdown('''5. Kemudian klik tombol proses. Jika gambar yang kamu 
        ambil adalah wajah maka akan muncul bentuk wajah kamu beserta rekomendasi potongan rambut''')
        st.markdown('   Selamat mencoba !?.')
# import opencv library
import cv2
import face_recognition
import glob
import sys

path = '/home/pooja/Desktop/FaceRecognition/images/*.jpg'  

files=glob.glob(path)
filelist = []
for file in files:
    #print(file)
    file = file.replace("/home/pooja/Desktop/FaceRecognition/images/","")
    file = file.replace(".jpg","")
    filelist.append(file)

#print filelist

if len(sys.argv) == 2:
    cam = cv2.VideoCapture(sys.argv[1])
# otherwise, capture video from live webcam
else:
    cam = cv2.VideoCapture(0)

image = []
encoding = []
for name in filelist:
    image_load=(face_recognition.load_image_file("/home/pooja/Desktop/FaceRecognition/images/"+name+".jpg"))
    tup = (image_load,name)
    image.append(tup)

for i,name in image:
    image_en = (face_recognition.face_encodings(i)[0])
    tup = (image_en,name)
    encoding.append(tup)


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while 1:
    # capture frame
    frame = cam.read()[1]
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)





    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        
        for face_encoding in face_encodings:
            for image, name1 in encoding:
                print (image)
                print (name1)
            # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces([image], face_encoding)
                name = "Unknown"

                if match[0]:
                    name = name1
                    #face_names.append(name)
                    break



            face_names.append(name)
            print (face_names)
    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        if name =="Unknown":
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0,128,0), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,128,0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	

# Release handle to the webcam
cam.release()
cv2.destroyAllWindows()
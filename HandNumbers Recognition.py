# Hand Numbers and Text Recognition
# Last Update 20220512

from djitellopy import Tello
from PIL import Image
import cv2, pytesseract

from datetime import datetime
from fingerCounter import fingerCounter

def countDown(frame_read, text, seconds, showCountDown):
    start_time = datetime.now()
    diff = (datetime.now() - start_time).seconds # converting into seconds
    while True:
        frame = frame_read.frame
        #frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_copy = frame.copy()
        im_pil = Image.fromarray(img)
        
        if showCountDown == True:
            cv2.putText(img_copy, text + str(seconds - diff), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:            
            cv2.putText(img_copy, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('tello hand', img_copy)
        diff = (datetime.now() - start_time).seconds
        if (diff > seconds):
            frame_read.stop
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   

    return

def identifyTextfromJpg(img):
    text="--"

    # Preprocessing the image starts
    # img = cv2.imread("sampleText.jpg")

    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area of the rectangle to be detected.
    # A smaller value like (10, 10) will detect each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    # Appplying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Creating a copy of image
    img_copy = img.copy()

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on to pytesseract for extracting text from it
    # Extracted text is then saved and coordinates of center returned
    max_area, max_x, max_y, max_w, max_h = 0, 0, 0, 0, 0
    centerX, centerY = 0, 0

    # Find biggest text
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        if area > max_area and area < 300*200:
            centerX = x + w//2
            centerY = y + h//2
            max_x = x
            max_y = y
            max_w = w
            max_h = h
            # Apply OCR on the cropped image
            #cropped = img_copy[y:y + h, x:x + w]
            #text = pytesseract.image_to_string(cropped, config="--psm 10")
            #cv2.putText(img_copy, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if max_w * max_h > 0:
        # Drawing a rectangle on copied image
        rect = cv2.rectangle(img_copy, (max_x, max_y), (max_x + max_w, max_y + max_h), (255, 255, 0), 2)

        # Cropping the text block for giving input to OCR
        cropped = img_copy[max_y:max_y + max_h, max_x:max_x + max_w]

        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped, config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789")
        text = ''.join(e for e in text if e.isalnum())
        cv2.putText(img_copy, text, (max_x+5, max_y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #imgStack = utils.stackImages(1, [img, img_copy, cropped])
    #imgStack = utils.stackImages(1, [img, img_copy])

    #cv2.imshow('tello ocr', imgStack)

    return text, centerX, centerY, max_w*max_y

# ———————————————–
# Main program
# ———————————————–

tello = Tello()
tello.connect()
tello.streamon()

# point to Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

m_fingerCounter = None
m_fingerCounter = fingerCounter()

myFrame = tello.get_frame_read()

#Buscar "Puño/Cero" para empezar
countDown(myFrame, "Give me a number, we start in ", 3, True)

cv2.destroyAllWindows()

countx = 0
lastnum = 0

#Buscar handnumber
while True:
    #myFrame = tello.get_frame_read()
    myf = myFrame.frame
    img = cv2.resize(myf, (640, 480))
    #img = cv2.resize(myf, (320, 240))

    _, _, count = m_fingerCounter.countFingers(img)
    # add the fingers in both hands
    num = count['RIGHT'] + count['LEFT']

    if num != lastnum:
        countx = 0

    if str(num) == "1" or str(num) == "2" or str(num) == "3" or str(num) == "4" or str(num) == "5":
        countx = countx + 1
        lastnum = num

    if (countx >= 100):
        myFrame.stop
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break 

#nos guardamos el ultimo
number = str(num)

print("You have chosen number: ", number,) 

cv2.destroyAllWindows()

countDown(myFrame, "You have chosen number " + number, 5, False)

countDown(myFrame, "Now find the number for the photo! We start in ", 3, True)

countx = 0

#Buscar predicción en texto
while True:
    frame = myFrame.frame

    #frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_copy = frame.copy()
    im_pil = Image.fromarray(img)
    cv2.imshow('tello hand', img_copy)

    text, centerX, centerY, area = identifyTextfromJpg(frame)

    if text == number:
        print("text on image detected:>", text, "<")
        countx = countx + 1

    if (countx == 3):
        myFrame.stop
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break    

#Hacer foto
cv2.imwrite("handnumberPhoto.jpg", img_copy)
cv2.imshow('tello photo', img_copy)
myFrame.stop

while True:
    frame = myFrame.frame

    frame = cv2.flip(frame, 1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_copy = frame.copy()
    im_pil = Image.fromarray(img)
    cv2.putText(img_copy, "Photo Taken, thanks!", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('tello hand', img_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break  

print ("finish")
###########
#Last Updated: 3/31/2026
#Author: Ethan
#Description: This is the OCR module for HealthTech. It uses EasyOCR to read text from images. The current implementation reads from a static image, but it can be modified to read from a video feed or camera input in the future.
###########

import time
import cv2
import easyocr

model_start_time = time.perf_counter()
ocr = easyocr.Reader(['en'], gpu=False) # this is the OCR reader, it takes a list of languages to read. In this case, it's set to English. It can be modified to read other languages if needed.
model_load_time = time.perf_counter()



def read_text_from_image(image):
    result = ocr.readtext(image)
    for (bbox, text, prob) in result:
        print(f"Text: {text}, Confidence: {prob}")
    return result



def main():
    print("Hello From HealthTech! \n")
    model_start_prediction_time = time.perf_counter()
    result = ocr.readtext(img)
    model_prediction_time = time.perf_counter()
    
    for (bbox, text, prob) in result:
        print(f"Text: {text}, Confidence: {prob}")
        ##gather info on bound box
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        ## make bound box
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        ## put text on image
        cv2.putText(img, text, (top_left[0]-20, top_left[1]), fontFace = cv2.FONT_ITALIC, fontScale = 0.5, color = (0, 0, 255), thickness = 1) #CV2 uses BGR

    
    print(f"\n Model load time: {model_load_time - model_start_time}s")
    print(f"Model prediction time: {model_prediction_time - model_start_prediction_time}s \n")

    print("New window opened...")
    cv2.imshow("OCR Result", img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread("TestImage/name.jpg")
    img = cv2.resize(img, (640,480)) # Resizethe image to 640x480, this actually improves accuracy and fits our needs

    if img is None:
        raise FileNotFoundError("signs.jpg not found")
    print("If you are waiting here forever, look for a new window with a iamge and press any key to continue")
    cv2.imshow("OCR Image", img) # I might choose a different method for this but for now yeah
    cv2.waitKey(0)# wait until a key is pressed
    cv2.destroyAllWindows() # close the window

    main()

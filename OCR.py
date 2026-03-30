###########
#Last Updated: 3/29/2026
#Author: Ethan
#Description: This is the OCR module for HealthTech. It uses PaddleOCR to read text from images. The current implementation reads from a static image, but it can be modified to read from a video feed or camera input in the future.
###########

#### USE THESE TO SAVE TIME, BUT IDK IF IT WORKS FOR OTHERS PROBABLY DON'T DO THIS FIRST RUN THROUGH ####
#import os
#os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
################################################################

from paddleocr import PaddleOCR
import cv2

img = cv2.imread(r"C:\Users\Ethan\Documents\GitHub\HealthTech\Test\signs.jpg")
img = cv2.resize(img, (640,480)) # Resize the image to 640x480, this actually improves accuracy and fits our needs
if img is None:
    raise FileNotFoundError("signs.jpg not found")
cv2.imshow("OCR Image", img) # I might choose a different method for this but for now yeah
cv2.waitKey(0)# wait until a key is pressed
cv2.destroyAllWindows() # close the window


ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en")


##This is where I learn that OCRs are just slow. It can probably handle one frame per second, but any more may be problematic with the current settings.
def main():
    print("Hello From HealthTech!")
    result = ocr.predict(img, text_rec_score_thresh = 0.7) ## THIS THRESH HOLD ALLOWS ALMOST EVERYTHING TO BE READ FROM THE SIGN BESIDES THE 7 FOR SOME REASON, SOME TWEAKING MAY NEED TO BE DONE
    for i, res in enumerate(result, start=1):
            for text, score in zip(res["rec_texts"], res["rec_scores"]): #res["thingy"] grabs whatever thingy is from the result. (EX: "rec_texts", "rec_boxes", "text_rec_scores_thresh")
                print(f"  - {text} ({score})")
            print()



if __name__ == "__main__":
    main()

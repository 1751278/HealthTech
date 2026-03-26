import easyocr

def main():
    print("Hello From HealthTech!")
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext('TestImage/indoorSigns.jpg', detail=0, paragraph=True)


if __name__ == "__main__":
    main()

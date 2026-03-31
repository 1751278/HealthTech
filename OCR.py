import easyocr

def main():
    print("Hello From HealthTech!")
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext('TestImage/indoorSigns.jpg', detail=0, paragraph=True)
    print(result)
    result = reader.readtext('TestImage/signs.jpg', detail=0, paragraph=True)
    print(result)
    result = reader.readtext('TestImage/name.jpg', detail=0, paragraph=True)
    print(result)


if __name__ == "__main__":
    main()

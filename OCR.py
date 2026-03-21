import easyocr

def main():
    print("Hello From HealthTech!")
    reader = easyocr.Reader(['en'])
    result = reader.readtext('Test/signs.jpg', detail=0, paragraph=True)
    print(result)


if __name__ == "__main__":
    main()

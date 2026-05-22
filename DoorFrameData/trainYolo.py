from ultralytics import YOLO

def main():
    # Load a pretrained model
    model = YOLO("yolo11n.pt") 

    # Train the model
    results = model.train(
        data="doorFrameData.yaml",   # path to your config file
        epochs=10,         # number of training rounds
        imgsz=640,          # input image size
        device=0            # use GPU 0 (or "cpu" if no GPU is available)
    )

if __name__ == '__main__':
    main()
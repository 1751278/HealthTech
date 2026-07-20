#################
# main.py
# Created by Kenshi Kadarusman February 28 2026
# Last Updated: ?
# Description: Just a simple main file. Doesn't do anything.
# TODO:
# - Integrate all modules into this main file and make them work together.
# - Create a user interface for the application.
#################
import cv2
import torch


# For NVIDIA GPUs
print(torch.cuda.is_available()) 


def main():
    print("Hello From HealthTech!")


if __name__ == "__main__":
    main()

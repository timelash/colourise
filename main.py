import cv2
import numpy as np
import sys
import os

def load_colorization_model():
    """Loads the pre-trained model for image colorization"""
    # Load the Caffe models provided by OpenCV for colorization
    proto_file = 'models/colorization_deploy_v2.prototxt'
    model_file = 'models/colorization_release_v2.caffemodel'
    pts_file = 'models/pts_in_hull.npy'

    net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
    
    # Load cluster centers used in the colorization process
    pts = np.load(pts_file)
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    return net

def colorize_image(image_path, output_path):
    """Colorizes the grayscale image using a pre-trained neural network"""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return

    # Convert the image to LAB color space
    h, w = img.shape[:2]
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = img_lab[:, :, 0]  # Extract the L channel (lightness)

    # Resize and normalize the L channel to fit the network's input size
    input_image = cv2.resize(l_channel, (224, 224))  # Model expects 224x224 input
    input_image = input_image.astype("float32") / 255.0  # Normalize between 0 and 1
    input_image -= 0.5  # Scale between -0.5 and 0.5
    input_image *= 2.0  # Adjust scaling

    # Load the model and pass the image through the network
    net = load_colorization_model()
    net.setInput(cv2.dnn.blobFromImage(input_image))
    ab_channels = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the ab channels to match the original image size
    ab_channels = cv2.resize(ab_channels, (w, h))

    # Combine the original L channel with the predicted ab channels
    img_lab_colorized = np.concatenate((l_channel[:, :, np.newaxis], ab_channels), axis=2)

    # Convert back to BGR color space
    img_bgr = cv2.cvtColor(img_lab_colorized, cv2.COLOR_LAB2BGR)

    # Clip any values that go outside the valid range and cast to uint8
    img_bgr = np.clip(img_bgr, 0, 255).astype('uint8')

    # Save the colorized image
    cv2.imwrite(output_path, img_bgr)
    print(f"Colorized image saved to {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python colorize.py <image_path>")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return

    output_path = "colorized_" + os.path.basename(image_path)
    colorize_image(image_path, output_path)

if __name__ == '__main__':
    main()

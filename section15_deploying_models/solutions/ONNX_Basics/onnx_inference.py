import cv2
from PIL import Image
import numpy as np
import onnxruntime
import json
import time


def crop_resize(image, new_size):
    # Get the dimensions of the original image
    width, height = image.size

    # Calculate the size of the square crop
    min_dim = min(width, height)

    # Calculate coordinates for the center crop
    left = (width - min_dim) // 2
    upper = (height - min_dim) // 2
    right = left + min_dim
    lower = upper + min_dim

    # Crop the image to a square
    square_image = image.crop((left, upper, right, lower))

    # Resize the image to the specified size
    resized_image = square_image.resize((new_size, new_size))

    return resized_image


def image_normalise_reshape(image, mean, std):
    h, w, c = image.shape
    image = image.transpose((2, 0, 1)) / 255

    np_means = np.array(mean).reshape(c, 1, 1)
    np_stds = np.array(std).reshape(c, 1, 1)

    norm_image = (image - np_means) / (np_stds + 1e-6)

    return np.expand_dims(norm_image, 0).astype(np.float32)


def run_sample(session, image):
    onnxruntime_input = {session.get_inputs()[0].name: image}
    onnxruntime_outputs = session.run(None, onnxruntime_input)
    class_index = np.argmax(onnxruntime_outputs[0])
    return class_index


if __name__ == "__main__":
    ort_session = onnxruntime.InferenceSession("./efficientnet_b1.onnx",
                                               providers=['CPUExecutionProvider'])
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    with open("../../data/imagenet_classes.json", "r") as file:
        img_net_classes = json.load(file)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()
    print("Press 'q' to quit, 'spacebar' to call the function.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.imshow('Camera', frame)

        # Wait for key press for 1 ms
        key = cv2.waitKey(5)

        if key & 0xFF == ord('q'):  # Exit if 'q' is pressed
            print("Exiting...")
            break
        elif key & 0xFF == ord(' '):  # Call function if spacebar is pressed
            cropped_frame = crop_resize(Image.fromarray(frame), 224)
            norm_image = image_normalise_reshape(np.array(cropped_frame), mean, std)

            start_time = time.time()
            image_class = run_sample(ort_session, norm_image)
            end_time = time.time()

            print("Class index:", image_class)
            print("Class Label:", img_net_classes[str(image_class)])
            print("Inference Time: %.4fs" % (end_time - start_time))

    cap.release()
    cv2.destroyAllWindows()

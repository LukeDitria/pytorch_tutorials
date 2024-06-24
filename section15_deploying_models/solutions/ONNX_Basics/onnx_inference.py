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
    # Get image dimensions (height, width, channels)
    h, w, c = image.shape

    # Move channel dimension to the front (assuming PyTorch format) and normalize pixel values by 255
    image = image.transpose((2, 0, 1)) / 255.0

    # Reshape mean and std into numpy arrays with proper dimensions for broadcasting
    np_means = np.array(mean).reshape(c, 1, 1)
    np_stds = np.array(std).reshape(c, 1, 1)

    # Normalize the image by subtracting the mean and dividing by the standard deviation (with epsilon for stability)
    norm_image = (image - np_means) / (np_stds + 1e-6)

    # Expand the dimension at index 0 to create a batch dimension (assuming batch size of 1)
    # and cast the data type to float32 for compatibility with most models
    return np.expand_dims(norm_image, 0).astype(np.float32)


def run_sample(session, image):
    # Prepare the input for the ONNX model
    onnxruntime_input = {session.get_inputs()[0].name: image}

    # Run the model and get the output
    onnxruntime_outputs = session.run(None, onnxruntime_input)

    # Get the class index with the highest score
    class_index = np.argmax(onnxruntime_outputs[0])

    return class_index


if __name__ == "__main__":
    # Initialize the ONNX runtime session with the specified model
    ort_session = onnxruntime.InferenceSession("./efficientnet_b1.onnx", providers=['CPUExecutionProvider'])

    # Define mean and standard deviation for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Load the class labels from a JSON file
    with open("../../data/imagenet_classes.json", "r") as file:
        img_net_classes = json.load(file)

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()
    print("Press 'q' to quit, 'spacebar' to call the function.")

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Wait for key press for 5 ms
        key = cv2.waitKey(5)

        if key & 0xFF == ord('q'):  # Exit if 'q' is pressed
            print("Exiting...")
            break
        elif key & 0xFF == ord(' '):  # Call function if space-bar is pressed
            # Process the captured frame
            cropped_frame = crop_resize(Image.fromarray(frame), 224)
            norm_image = image_normalise_reshape(np.array(cropped_frame), mean, std)

            # Measure the inference time
            start_time = time.time()
            image_class = run_sample(ort_session, norm_image)
            end_time = time.time()

            # Display the classification result and inference time
            print("Class index:", image_class)
            print("Class Label:", img_net_classes[str(image_class)])
            print("Inference Time: %.4fs" % (end_time - start_time))

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
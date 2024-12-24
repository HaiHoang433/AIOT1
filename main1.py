import cv2
import pandas as pd
import dlib
import numpy as np
import os
import time
from pyzbar.pyzbar import decode
from PIL import Image, ImageEnhance

# Define IP camera URL
IP_CAMERA_URL = "https://192.168.1.6:8080/video"

# Ensure Faces folder exists
if not os.path.exists("Faces"):
    os.makedirs("Faces")

# Function to verify the security of the QR code content
def is_secure_qr_code(data):
    secure_prefix = "SECURE_"
    return data.startswith(secure_prefix)

# Function to enhance brightness of an image
def lighten_image(image, factor=1.5):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(gray_image)
    enhancer = ImageEnhance.Brightness(img)
    brightened_img = enhancer.enhance(factor)
    return np.array(brightened_img)

# Function to decode the QR code
def detect_secure_qr_code(image):
    brightened_image = lighten_image(image, factor=1.5)
    decoded_objects = decode(brightened_image)
    result = {"qr_codes": []}
    
    for obj in decoded_objects:
        qr_data = obj.data.decode("utf-8")
        is_secure = is_secure_qr_code(qr_data)
        result["qr_codes"].append({
            "data": qr_data,
            "is_secure": is_secure
        })
    
    if not result["qr_codes"]:
        result["message"] = "No QR code detected in the image."
    
    return result

# Function to load existing QR codes from Excel to check for duplicates
def load_existing_qr_data(excel_path="qr_code_data1.xlsx"):
    try:
        df = pd.read_excel(excel_path)
        return df["Số"].tolist()
    except FileNotFoundError:
        return []

# Function to save QR code data to an Excel file
def save_qr_data_to_excel(data, excel_path="qr_code_data1.xlsx"):
    columns = ["ID", "Name", "Birthdate", "Gender", "Address", "Issue Date"]
    id_part, other_data = data.split('||')
    qr_data = [id_part] + other_data.split('|')
    birthdate = f"{qr_data[2][:2]}/{qr_data[2][2:4]}/{qr_data[2][4:]}"
    issue_date = f"{qr_data[5][:2]}/{qr_data[5][2:4]}/{qr_data[5][4:]}"
    data_dict = {
        "Số": qr_data[0],
        "Họ và tên": qr_data[1],
        "Ngày sinh": birthdate,
        "Giới tính": qr_data[3],
        "Nơi thường trú": qr_data[4],
        "Ngày tạo CCCD": issue_date
    }
    df = pd.DataFrame([data_dict])
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    return {"message": "QR code data saved to Excel successfully.", "file_path": excel_path}

# Function to detect and save the best face
def detect_and_save_best_face(input_image_path, output_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Could not load image.")
        return None
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No faces detected.")
        return None
    
    best_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = best_face
    cropped_face = image[y:y+h, x:x+w]
    cv2.imwrite(output_path, cropped_face)
    print(f"Best face saved at {output_path}")
    return output_path

# Function to compute face descriptors
def compute_face_descriptor(image, face_detector, shape_predictor, face_rec_model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces) == 0:
        return None, []
    
    face_descriptors = []
    for face in faces:
        shape = shape_predictor(gray, face)
        face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
        face_descriptors.append(np.array(face_descriptor))
    
    return face_descriptors, faces

# Function to compare two face descriptors
def compare_faces(descriptor1, descriptor2):
    distance = np.linalg.norm(descriptor1 - descriptor2)
    return distance

# Main function for face comparison and saving the best face
def main(static_image_path, id):
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    
    static_image = cv2.imread(static_image_path)
    static_face_descriptors, _ = compute_face_descriptor(static_image, face_detector, shape_predictor, face_rec_model)
    if not static_face_descriptors:
        print("No faces found in the static image.")
        return
    
    static_descriptor = static_face_descriptors[0]
    video_capture = cv2.VideoCapture(IP_CAMERA_URL)
    print("Capturing faces for 10 seconds...")

    best_similarity = 0
    best_face_image = None

    start_time = time.time()
    while time.time() - start_time < 10:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        live_face_descriptors, faces = compute_face_descriptor(frame, face_detector, shape_predictor, face_rec_model)
        if live_face_descriptors:
            for live_descriptor, face in zip(live_face_descriptors, faces):
                distance = compare_faces(static_descriptor, live_descriptor)
                similarity = 1 - min(distance / 1.5, 1.0)
                similarity_percentage = int(similarity * 100)
                
                if similarity_percentage > best_similarity:
                    best_similarity = similarity_percentage
                    best_face_image = frame[face.top():face.bottom(), face.left():face.right()]

    if best_face_image is not None:
        real_face_path = f"Faces/{id}-RealFace.png"
        cv2.imwrite(real_face_path, best_face_image)
        print(f"Best face saved as {real_face_path} with similarity {best_similarity}%")
    else:
        print("No face detected with significant similarity.")

    video_capture.release()
    cv2.destroyAllWindows()

# Updated capture and process function
def capture_and_process_qr_codes():
    cap = cv2.VideoCapture(IP_CAMERA_URL)
    existing_codes = load_existing_qr_data()
    
    while True:
        print("Waiting for the next person...")
        last_qr_code = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from the IP camera. Check the connection.")
                break

            result = detect_secure_qr_code(frame)
            if "qr_codes" in result and result["qr_codes"]:
                qr_data = result["qr_codes"][0]["data"]
                current_id = qr_data.split('||')[0]
                
                if current_id in existing_codes:
                    print(f"Duplicate ID {current_id} detected, skipping...")
                    last_qr_code = qr_data
                else:
                    print("QR Code Detected:", qr_data)
                    save_result = save_qr_data_to_excel(qr_data)
                    print(save_result["message"])

                    # Capture the CCCD image
                    print("Now capture the whole CCCD. Press 'Space' to capture.")
                    while True:
                        ret, frame = cap.read()
                        cv2.imshow("Capture CCCD", frame)
                        if cv2.waitKey(1) & 0xFF == ord(' '):
                            cccd_image_path = f"Faces/{current_id}-CCCD.png"
                            cv2.imwrite(cccd_image_path, frame)
                            print(f"CCCD image saved at {cccd_image_path}")

                            # Detect and save the best face
                            face_image_path = f"Faces/{current_id}-Face.png"
                            face_result = detect_and_save_best_face(cccd_image_path, face_image_path)
                            if face_result:
                                print(f"Face image saved at {face_image_path}")
                                
                                # Run the face comparison and saving
                                main(f"Faces/{current_id}.png", current_id)
                            break
                        elif cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    last_qr_code = qr_data
                    existing_codes.append(current_id)
                    print("Processing completed for this person.")
                    print("Next Person...")
                    break  # Move to the next person

            cv2.imshow("QR Code Scanner", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the updated function
capture_and_process_qr_codes()
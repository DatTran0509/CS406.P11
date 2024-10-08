import streamlit as st
import cv2
import numpy as np
import os
import random
from scipy.spatial.distance import euclidean
from PIL import Image

def calculate_histogram(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_hist = cv2.calcHist([rgb_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    rgb_hist = cv2.normalize(rgb_hist, rgb_hist).flatten()
    return rgb_hist

def load_dataset(dataset_folder):
    image_histograms = {}
    for root, _, files in os.walk(dataset_folder):
        for image_file in files:
            if image_file.endswith('.jpg'):  
                image_path = os.path.join(root, image_file)
                image = cv2.imread(image_path)
                rgb_hist = calculate_histogram(image)
                image_histograms[image_path] = rgb_hist

    return image_histograms

def match_images(query_hist, image_histograms, top_n=10):
    # Tạo một danh sách để lưu khoảng cách Euclide giữa query_hist và các histogram khác
    distances = []
    # Duyệt qua các histogram của các ảnh trong image_histograms
    for image_path, hist in image_histograms.items():
        # Tính khoảng cách Euclide giữa query_hist và hist hiện tại
        dist = euclidean(query_hist, hist)
        distances.append((image_path, dist))
    
    # Sắp xếp danh sách khoảng cách theo thứ tự tăng dần
    distances.sort(key=lambda x: x[1])
    
    # Trả về danh sách top_n hình ảnh có khoảng cách nhỏ nhất (hình ảnh tương tự nhất)
    return distances[:top_n]
def main():
    st.title("Image Similarity Based on HSV Histogram Analysis")
    # Load dataset
    dataset_folder = r"./22520236_Lab-02/dataset/seg" 
    image_histograms = load_dataset(dataset_folder)
    
    # Nhập số lượng hình ảnh tương tự cần tìm
    top_n = st.number_input("Enter the number of top similar images (from 1 to 20):", min_value=1, max_value=20, value=10)
    
    # Tải ảnh lên
    uploaded_image = st.file_uploader("Upload an image from seg_test", type=["jpg"])
    
        # Kiểm tra xem có tệp nào được tải lên không
    if uploaded_image is not None:
            input_image = Image.open(uploaded_image)
            input_image_np = np.array(input_image)  
            
            # Xử lý ảnh nếu là ảnh xám
            if len(input_image_np.shape) == 2 or input_image_np.shape[2] == 1:
                input_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_GRAY2BGR)

            # Tính histogram của ảnh đã tải lên
            input_rgb_hist = calculate_histogram(input_image_np)
            similar_images = match_images(input_rgb_hist, image_histograms, top_n=top_n)

            col1, col2 = st.columns(2)
        
            with col1:
                st.write(f"Your Image:")
                st.image(input_image, caption="Uploaded Image", use_column_width=True)

            with col2:   
                st.write(f"Top {top_n} similar images:")
                for i in range(0, len(similar_images), 3):
                    cols = st.columns(3)  
                    for j, (img_path, dist) in enumerate(similar_images[i:i+3]):
                        with cols[j]:
                            st.image(img_path, caption=f"{os.path.basename(img_path)}", use_column_width=True)

if __name__ == "__main__":  
    main()


# **CancerDetect AI – AI-Based Cancer Detection System**

## **Overview**
CancerDetect AI is an **AI-powered web-based system** designed to detect cancerous tissues from medical images. The system allows users to **upload medical scans** for analysis and classification using **Deep Learning**. The results are displayed through an **interactive web interface**, with predictions stored securely in **MongoDB** for further review.

Additionally, a **demonstration video** showcasing the project is available, providing an in-depth look at its features and functionality.

### **Why is this project important?**
- Enables **early cancer detection**, improving patient survival rates.
- Reduces the **dependency on manual diagnosis**, which can be time-consuming and error-prone.
- Provides a **fast and efficient automated diagnostic tool** for healthcare providers.
- Stores predictions and patient data in **MongoDB**, allowing easy access and analysis.
- Available as a **fully interactive web application** for easy usability.

---

## **Data and Features**
The system processes medical image datasets and extracts key features using **Convolutional Neural Networks (CNNs)** and other deep learning architectures.

### **Dataset Structure**
| Column Name          | Description |
|----------------------|------------|
| **patient_id**      | Unique identifier for each patient. |
| **image_data**      | Raw pixel data of the medical scan. |
| **diagnosis**       | Label indicating if cancer is present (Positive/Negative). |
| **scan_type**       | Type of scan (e.g., MRI, X-ray, Mammogram). |
| **prediction**      | Model's classification result. |

---

## **How It Works**
CancerDetect AI follows a structured pipeline for image-based cancer detection:

1. **Image Upload & Preprocessing**
   - The user uploads a medical scan via the web interface.
   - The image is resized and normalized for model compatibility.
   - Convert the image into a numerical format for AI processing.

2. **Feature Extraction & Classification**
   - Use **CNNs (Convolutional Neural Networks)** for feature extraction.
   - Predict whether the image contains cancerous tissues.
   - Two models are used for prediction: **Model 1 (32x32 input) and Model 2 (8x8 input)**.

3. **Result Storage & Visualization**
   - Prediction results are stored in **MongoDB**.
   - If cancer is detected, the system redirects to a detailed alert page.
   - If no cancer is found, a confirmation page is displayed.
   - The web interface presents the analysis with a user-friendly experience.

4. **Project Video**
   - A video demonstrating the functionality and features of CancerDetect AI is available for users to understand how the system works in real-time.

---

## **Models Used**
The project leverages multiple **Deep Learning models** for image classification:

### **Baseline Model: Convolutional Neural Networks (CNNs)**
- **Model 1:** Trained on **32x32** resized images for high accuracy.
- **Model 2:** Trained on **8x8** images for lightweight performance.
- Both models utilize **TensorFlow/Keras** with optimized layers for medical image recognition.

### **Performance Metrics Used**
The model's effectiveness is measured using:
- **Accuracy** – Measures overall correct classifications.
- **Precision & Recall** – Ensures reliable identification of cancerous tissues.
- **Confusion Matrix** – Helps analyze false positives and false negatives.

---

## **Technologies Used**
CancerDetect AI integrates several technologies and frameworks for efficient processing:
- **FastAPI** – Backend framework for API development.
- **TensorFlow/Keras** – Deep learning framework for image classification.
- **Pillow & OpenCV** – Image processing libraries.
- **MongoDB** – NoSQL database for storing model predictions.
- **HTML, Tailwind CSS, JavaScript** – Frontend technologies for an interactive web experience.

---

## **Future Enhancements**
- **Enhancing Model Accuracy** – Implementing **Vision Transformers (ViTs)** for improved classification.
- **Real-time Detection** – Developing a live scanning feature.
- **Multi-class Classification** – Detecting different types of cancer.
- **Federated Learning Support** – Ensuring privacy by training on decentralized datasets.
- **Advanced Web Features** – Improving UI/UX for better usability.

---

## **Conclusion**
CancerDetect AI provides a **reliable, AI-driven solution** for detecting cancer in medical images. By combining **Deep Learning, FastAPI, MongoDB, and a responsive web interface**, the system delivers fast, accurate, and scalable medical image analysis. With further enhancements, CancerDetect AI has the potential to **revolutionize early cancer detection**, aiding both patients and healthcare professionals in improving diagnosis outcomes.

A **detailed video** explaining the system's features and working is available for reference.

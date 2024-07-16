from keras.preprocessing import image
from keras.models import load_model
import numpy as np

# Fungsi untuk memuat model dan memprediksi gambar
def predict(img_path , model_path, target_size) :
    model = load_model(model_path)
    # Memuat dan memproses gambar
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi
    # Prediksi
    prediction = model.predict(img_array)
    # Mendekode prediksi
    predicted_class = 'colon benign tissue' if prediction[0] > 0.5 else 'colon adenocarcinoma'
    confidence = prediction[0][0] * 100  # Konversi ke persentase
    return predicted_class, confidence


def inceptionV3(img_path):
    # Memuat model
    model_path = "ml/InceptionV3_model_best.h5"  # Ganti dengan path ke model yang anda gunakan
    result  = predict(img_path, model_path, (299,299))
    return result


def vgg19(img_path):
    # Memuat model
    model_path = "ml/vgg19_model_best.h5"  # Ganti dengan path ke model yang anda gunakan
    # Ganti dengan path ke model yang anda gunakan
    result  = predict(img_path, model_path, (224, 224))
    return result

# hasil =  inceptionV3("../colonn75.jpeg")
# print("hasil : ",hasil)
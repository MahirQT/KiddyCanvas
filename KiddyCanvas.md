
# 🎨 KiddyCanvas

**KiddyCanvas** is a web-based, AI-powered educational app designed for children to learn alphabets and numbers by drawing them — just like on a digital slate! Kids draw letters or digits, and the app recognizes them using a deep learning model and speaks the result out loud.

---

## ✨ Features

- 🖍️ **Interactive Drawing Canvas** – Web-based canvas for freehand drawing
- 🤖 **AI Character Recognition** – Powered by a CNN trained on the EMNIST dataset
- 🔊 **Voice Output** – Uses `pyttsx3` to speak predictions like “A for Apple”
- 🎨 **Kid-Friendly UI** – Playful colors, rounded fonts, and simple buttons
- 🌐 **Works in Browser** – Built using HTML, CSS, JavaScript, and Flask

---

## 🧠 Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (Canvas API)
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras CNN
- **Text-to-Speech**: pyttsx3
- **Image Processing**: Pillow (PIL)
- **Dataset**: [EMNIST Balanced](https://www.nist.gov/itl/products-and-services/emnist-dataset)

---

## 🗂️ Project Structure

```
KiddyCanvas/
├── app.py                  # Flask backend
├── cnn_emnist_model.h5     # Trained CNN model
├── label_mapping.npy       # Label map (index → character)
├── requirements.txt        # All Python dependencies
├── render.yaml             # Optional: Render deployment file
├── static/
│   ├── style.css           # CSS for UI styling
│   └── script.js           # Canvas & API call handling
├── templates/
│   └── index.html          # Main UI page
└── README.md               # You're reading it!
```

---

## 🚀 How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/MahirQT/KiddyCanvas.git
cd KiddyCanvas
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate       # On Windows
# or
source venv/bin/activate      # On Mac/Linux
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 📢 References

- [EMNIST Dataset – NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [TensorFlow MNIST CNN](https://www.tensorflow.org/tutorials/quickstart/advanced)
- [UNICEF – Early Learning](https://www.unicef.org/early-childhood-development)
- [pyttsx3 TTS](https://pypi.org/project/pyttsx3/)

---

## 📈 Future Scope

- 🌍 Multilingual voice output
- 🏆 Gamified learning modes
- ✍️ Real-time handwriting improvement feedback
- 📱 Mobile responsive design
- 🧩 Support for full words & phonics

---

## 🙌 Credits

Created with ❤️ by [MahirQT](https://github.com/MahirQT)  
Logo and UI designed to inspire curiosity and fun!

---

## 🧸 Let's make learning fun, one doodle at a time!

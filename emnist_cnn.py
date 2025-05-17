import numpy as np
import gzip
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, ImageOps, ImageTk
import tkinter as tk
import joblib
from tkinter import ttk

def load_idx(filename):
    """Load IDX formatted EMNIST data."""
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_items = int.from_bytes(f.read(4), byteorder='big')
        shape = [num_items]
        if magic == 2051:
            rows = int.from_bytes(f.read(4), byteorder='big')
            cols = int.from_bytes(f.read(4), byteorder='big')
            shape.extend([rows, cols])
        elif magic == 2049:
            pass
        else:
            raise ValueError(f"Unknown magic number: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
        return data

def load_and_preprocess_data():
    """Load and preprocess EMNIST data."""
    folder = "emnist_balanced"
    X_train = load_idx(f"{folder}/emnist-balanced-train-images-idx3-ubyte.gz")
    y_train = load_idx(f"{folder}/emnist-balanced-train-labels-idx1-ubyte.gz")
    X_test = load_idx(f"{folder}/emnist-balanced-test-images-idx3-ubyte.gz")
    y_test = load_idx(f"{folder}/emnist-balanced-test-labels-idx1-ubyte.gz")

    # Reshape for CNN input (N, 28, 28, 1) and normalize
    X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Convert labels to one-hot vectors
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    return X_train, y_train_cat, X_test, y_test_cat

def build_model(num_classes):
    """Build and compile the CNN model."""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

class DrawApp:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("EMNIST Drawing Recognition")
        self.window.configure(bg='#f0f0f0')
        
        # Create main frame
        self.main_frame = ttk.Frame(self.window, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frames for layout
        self.drawing_frame = ttk.LabelFrame(self.main_frame, text="Drawing Area", padding="5")
        self.drawing_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.preview_frame = ttk.LabelFrame(self.main_frame, text="Processed Image", padding="5")
        self.preview_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.control_frame = ttk.Frame(self.main_frame, padding="5")
        self.control_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")
        
        # Create canvas with border
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(
            self.drawing_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg='white',
            highlightthickness=1,
            highlightbackground='#cccccc'
        )
        self.canvas.pack(padx=5, pady=5)
        
        # Create processed image preview
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack(padx=5, pady=5)
        
        # Initialize image
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # Create buttons with styling
        style = ttk.Style()
        style.configure('Action.TButton', font=('Arial', 12))
        
        self.predict_btn = ttk.Button(
            self.control_frame,
            text="Predict",
            command=self.predict,
            style='Action.TButton'
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(
            self.control_frame,
            text="Clear",
            command=self.clear,
            style='Action.TButton'
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Create prediction frame
        self.prediction_frame = ttk.LabelFrame(self.main_frame, text="Prediction", padding="5")
        self.prediction_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        
        self.prediction_label = ttk.Label(
            self.prediction_frame,
            text="Draw a character to predict",
            font=('Arial', 24)
        )
        self.prediction_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(
            self.prediction_frame,
            text="",
            font=('Arial', 12)
        )
        self.confidence_label.pack(pady=2)
        
        # Add drawing settings
        self.settings_frame = ttk.LabelFrame(self.main_frame, text="Drawing Settings", padding="5")
        self.settings_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")
        
        # Brush size slider
        self.brush_size = tk.IntVar(value=8)
        ttk.Label(self.settings_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        self.brush_slider = ttk.Scale(
            self.settings_frame,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.brush_size,
            length=200
        )
        self.brush_slider.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        
        # Bind events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.update_preview)
        
        # Add keyboard shortcuts
        self.window.bind("<Control-p>", lambda e: self.predict())
        self.window.bind("<Control-c>", lambda e: self.clear())
        self.window.bind("<Control-z>", lambda e: self.clear())
        
        # Add status bar
        self.status_bar = ttk.Label(
            self.window,
            text="Draw a character and click Predict (Ctrl+P) | Clear: Ctrl+C or Ctrl+Z",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.window.mainloop()
    
    def paint(self, event):
        x1, y1 = (event.x - self.brush_size.get()), (event.y - self.brush_size.get())
        x2, y2 = (event.x + self.brush_size.get()), (event.y + self.brush_size.get())
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill='black')
    
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.preview_label.configure(image="")
        self.prediction_label.configure(text="Draw a character to predict")
        self.confidence_label.configure(text="")
    
    def update_preview(self, event=None):
        # Update the preview with the current drawing
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        preview_img = img.resize((280, 280), Image.Resampling.NEAREST)
        preview_tk = ImageTk.PhotoImage(image=preview_img)
        self.preview_label.imgtk = preview_tk
        self.preview_label.configure(image=preview_tk)
    
    def predict(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img).astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)
        
        prediction = self.model.predict(img)
        label = np.argmax(prediction)
        confidence = prediction[0][label] * 100
        
        # Get the character from the mapping
        character = LABEL_MAPPING.get(label, "Unknown")
        
        # Update prediction label with character
        self.prediction_label.configure(text=f"Predicted: {character}")
        
        # Update confidence with color based on threshold
        confidence_text = f"Confidence: {confidence:.1f}%"
        if confidence >= 90:
            self.confidence_label.configure(text=confidence_text, foreground="green")
        elif confidence >= 70:
            self.confidence_label.configure(text=confidence_text, foreground="orange")
        else:
            self.confidence_label.configure(text=confidence_text, foreground="red")

def main():
    # Load and preprocess data
    X_train, y_train_cat, X_test, y_test_cat = load_and_preprocess_data()
    
    # Build and train model
    model = build_model(y_train_cat.shape[1])
    
    # Train on entire dataset with validation split
    print("Training on entire dataset...")
    model.fit(
        X_train, 
        y_train_cat, 
        epochs=20,  # Increased epochs
        batch_size=64,  # Smaller batch size for better learning
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save("cnn_emnist_model.h5")
    np.save("label_mapping.npy", y_train_cat)
    print("✅ Model saved as cnn_emnist_model.h5")
    
    # Load model and start drawing app
    model = load_model("cnn_emnist_model.h5")
    DrawApp(model)

if __name__ == "__main__":
    main() 
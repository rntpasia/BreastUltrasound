# Breast Ultrasound AI — YOLOv12 Streamlit App

Instance segmentation and classification of breast ultrasound images using YOLOv12.

## 📁 Project Structure

```
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level apt packages (for Streamlit Cloud)
├── .streamlit/
│   └── config.toml         # Theme & server config
└── .gitignore
```

## 🚀 Deploy to Streamlit Cloud

1. Push this folder to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo, branch `main`, and main file `app.py`
4. Click **Deploy** — done!

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## 🔬 How to Use

1. **Upload your weights** — use the sidebar to upload your trained `best.pt` file from:
   `runs/segment/train/weights/best.pt`
2. **Upload an ultrasound image** — JPG or PNG format
3. **Adjust confidence threshold** in the sidebar if needed
4. Click **Run Analysis** to see segmentation masks, bounding boxes, and class predictions

## ⚠️ Notes

- `flash-attn` and `onnxruntime-gpu` are **excluded** from `requirements.txt` because Streamlit Cloud runs on CPU. The model will still work — just without GPU acceleration.
- For GPU inference, deploy on [Railway](https://railway.app), [Hugging Face Spaces (GPU)](https://huggingface.co/spaces), or a cloud VM.
- Model weights (`.pt` files) are uploaded at runtime via the sidebar — they are not committed to the repo.

## Classes

| Label | Description |
|---|---|
| `benign` | Non-cancerous mass |
| `malignant` | Cancerous mass |
| `normal` | No abnormality detected |

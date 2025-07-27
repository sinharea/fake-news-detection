# Deepfake News Detection System 🧠🔍

**AI-powered system to detect deepfake and fake news using machine learning and computer vision**  
*Demo featured on LinkedIn with video walkthrough*

---

## 🚀 Overview

The **Deepfake News Detection System** is a comprehensive tool built to spot both synthetic video (deepfakes) and misleading text (fake news).  
Leveraging Python, computer vision, and NLP-based classifiers, this project ensures media authenticity and helps combat misinformation effectively.  
The demo video hosted on LinkedIn showcases system interface, detection confidence, and real-world performance.

---

## 🎥 Demo Video

[▶️ Watch the live demo on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7336392761912315905/)  
*See how the model evaluates content and presents confidence-based verdicts in real time.*

---

## 🧠 How It Works

- **Media Preprocessing**: Video frames and news text are extracted and cleaned  
- **Feature Extraction**: Visual frames are analyzed using image processing techniques, while text is vectorized with NLP pipelines  
- **Classification Pipeline**:
  - Detect video manipulation using pretrained image model
  - Classify textual content with text classification model
- **Output**: Provides a confidence score and final verdict—Real or Fake

---

## 📁 Project Structure

fake-news-detection/
├── Untitled5_Updated (2).ipynb ← Main Jupyter notebook
├── data/
│ ├── test_videos/ ← Deepfake / real samples
│ └── news_texts.csv ← News text samples for classification
├── outputs/
│ ├── evaluation_metrics.json
│ ├── confusion_matrix.png
│ └── sample_predictions.csv
└── README.md


---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+  
- Jupyter Notebook environment  
- Libraries: `numpy`, `pandas`, `scikit-learn`, `opencv-python`, `tensorflow` or `torch` depending on your model

### Setup Steps

```bash
git clone https://github.com/sinharea/fake-news-detection.git
cd fake-news-detection
```

# Optional: create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt  # install dependencies
jupyter notebook Untitled5_Updated\ \(2\).ipynb
```

## 🕹️ Usage Instructions:
  - **💻 Load:** Open the notebook in Jupyter or Google Colab
  - **🔄 Run Cells:** Execute each cell in sequence
  - **🧠 Train or Load Model:** Either train from scratch or use saved weights
  - **📈 Evaluate:** Review confusion matrix, F1 score, etc.
  - **🎥 Test:** Input custom news or media for prediction

## 📊 Evaluation and Results:
  - **📏 Accuracy, Precision, Recall, F1-Score**
  - **📊 Confusion Matrix** saved as image
  - **📁 Outputs Folder:** Includes metrics, plots, predictions

## 🌟 Features and Highlights:
  - 🧬 Multi-modal: Supports text and video input
  - 🛡️ Deep Learning: CNN + NLP pipelines
  - 📱 Simple UI: Score prediction with confidence bar
  - 🔍 Visual feedback & model transparency

## 📚 References and Research Context:
  - https://thesai.org/Downloads/Volume14No1/Paper_44-A_Novel_Smart_Deepfake_Video_Detection_System.pdf
  - https://arxiv.org/abs/2505.06796
  - https://github.com/enricollen/DeepfakeDetection
  - https://en.wikipedia.org/wiki/Deepfake_pornography
  - https://www.theguardian.com/us-news/article/2024/jun/07/how-to-spot-a-deepfake

## 🙌 Contributions Welcome:
  - **🤝 How to contribute:** Fork, star, submit PRs
  - **📐 Guidelines:** Use PEP8, clear comments, and readable structure
  - **🎯 Testing:** Test code before pushing
  - **📄 Notebook:** Must include markdown explanations for new additions

## 📩 License and Contact:
  - **👤 Author:** Rea Sinha
  - **🏫 Institute:** Indian Institute of Information Technology, Guwahati
  - **🔗 LinkedIn Demo:** https://www.linkedin.com/feed/update/urn:li:activity:7336392761912315905/
  - **🪪 License:** MIT — free to use with attribution

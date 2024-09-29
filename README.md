<!-- ABOUT THE PROJECT -->
## About The Project

Ever wonder what your baby’s cries really mean? This project classifies the sounds of baby cries into categories such as belly pain, burping, discomfort, hunger, and tiredness. Utilizing the [Infant Cry Audio Corpus dataset](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus), the model helps parents better understand their baby's needs.

Audio data is preprocessed using TensorFlow to squeeze audio dimensions and create spectrograms, which visually represent sound. The data is divided into training, validation, and test sets for efficient training. The model features convolutional layers for feature extraction, followed by fully connected layers for classification. Deployed with Streamlit, the application allows users to upload audio files or input live recordings in WAV format. Once processed, it displays the classification of the audio, assisting parents in decoding the meaning behind their baby’s cries.

![View project](resources/view.png)

This innovative tool aids in understanding a baby’s needs effectively!

### Built With

* [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
* [![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
* [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
* [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
* [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
* [![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)](https://scipy.org/)



<!-- GETTING STARTED -->
## Getting Started

### Run Locally

1. Clone the repo
2. Install necessary packages
   ```sh
   pip install -r requirements.txt
   ```
4. Run streamlit_app.py 
   ```js
   streamlit run streamlit_app.py
   ```



<!-- DEMO -->
## Demo

![Demo project](resources/demo.gif)

import streamlit as st
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from st_audiorec import st_audiorec
import io, pathlib

# Load the saved model
model = tf.keras.models.load_model('cry_classification_model_scenario_1.h5')

# Label names
FILE_PATH = './donateacry_corpus'
dataset = pathlib.Path(FILE_PATH)
cry_list = np.array(tf.io.gfile.listdir(str(dataset)))
label_names = cry_list

def preprocess_wav(audio_data):
    audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    audio_tensor = audio_tensor / 32768.0
    pad_size = tf.maximum(0, 16000 - tf.shape(audio_tensor)[0])
    audio_tensor = tf.pad(audio_tensor, [[0, pad_size]], mode='CONSTANT')
    audio_tensor = tf.slice(audio_tensor, [0], [16000])  
    spectrogram = tf.signal.stft(
        signals=audio_tensor,
        frame_length=255,
        frame_step=128
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return tf.expand_dims(spectrogram, axis=0)

def predict_wav(audio_data):
    if isinstance(audio_data, bytes):
        # Handle uploaded file
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
    elif isinstance(audio_data, np.ndarray):
        # Handle NumPy array
        audio_np = audio_data
    else:
        raise ValueError("Unsupported audio data type")

    # Preprocess the audio data and predict the label
    spectrogram = preprocess_wav(audio_np)
    prediction = model.predict(spectrogram)
    predicted_label_idx = tf.argmax(prediction, axis=1).numpy()[0]
    predicted_label = label_names[predicted_label_idx]
    return predicted_label

@st.experimental_dialog('Result', width="small")
def show_label(label):
    if label == 'belly_pain':
        st.subheader('Feeling Some Tummy Discomfort', divider='red')
        st.image('./asset/belly_pain_1.jpeg', use_column_width='always')
    elif label == 'burping':
        st.subheader('A Little Burping Time!', divider='red')
        st.image('./asset/burping.jpeg', use_column_width='always')
    elif label == 'discomfort':
        st.subheader('Feeling a Bit Uncomfortable', divider='red')
        st.image('./asset/discomfort_1.jpeg', use_column_width='always')
    elif label == 'hungry':
        st.subheader('Time for a Yummy Snack!', divider='red')
        st.image('./asset/hungry.jpeg', use_column_width='always')
    elif label == 'tired':
        st.subheader('Getting Sleepy and Cozy', divider='red')
        st.image('./asset/tired.jpeg', use_column_width='always')

def main():
    st.title('_Infant Cry Classification_')
    
    tab1, tab2 = st.tabs(["Audio Recorder", "Upload Audio file"])

    predicted_label = None
    with tab1:
        wav_audio_data = st_audiorec()
        if wav_audio_data is not None:
            audio_np = np.frombuffer(wav_audio_data, dtype=np.int16)
            predicted_label = predict_wav(audio_np)
            # st.write(f'The predicted label: {predicted_label}')
            if predicted_label is not None:
                # st.write(predicted_label)
                submit_record = st.button("Submit", key='record_input')
                if submit_record:
                    show_label(predicted_label)

    with tab2:
        wav_audio_data = st.file_uploader('Upload wav file', type='wav')
        if wav_audio_data is not None:
            bytes_data = wav_audio_data.getvalue()
            predicted_label = predict_wav(bytes_data)
            # st.write(f'The predicted label: {predicted_label}')
            if predicted_label is not None:
                # st.write(predicted_label)
                submit_file = st.button("Submit", key='file_input')
                if submit_file:
                    show_label(predicted_label)

if __name__ == '__main__':
    main()

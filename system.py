# feature extractoring and preprocessing data
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import soundfile as sf
import tempfile
import os
import signal as sig
import noisereduce as nr
import glob 
import scipy.signal as sig
import torch 
from torch import nn 
import pandas  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split



# Load dataset files
ipynb_background_noise_detection_module_path = "C:\\Users\\Fatima Ansari\\Desktop\\oice\\voice\\background-noise-detection.ipynb"
def load_background_noise_detection():
    # Your function code here
    pass

# Calling the function
interpreter = load_background_noise_detection()

# Example usage of glob to find all .txt files in a folder
files = glob.glob('./data_folder/*.txt')
print(files)

class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=76160, out_features=64, bias=True)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(in_features=64, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if self.linear1 == module:
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        if self.linear2 == module:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        logits = self.linear2(x)
        prediction = self.sigmoid(logits).squeeze(1)

        prediction[prediction>=0.5] = 1
        prediction[prediction<0.5] = 0

        prediction = prediction.to(dtype=torch.int64)

        return prediction

# Initialize session state variables
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = None
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None

st.set_page_config(page_title="Noise Reduction System Using CNN", page_icon="🎧", layout="wide")

def apply_noise_reduction(audio_signal, sr, volume_increase_factor=1.5):
    """Apply noise reduction with adjusted filtering, normalization, and volume increase"""
    try:
        # Apply noise reduction (less aggressive)
        reduced_noise = nr.reduce_noise(
            y=audio_signal,
            sr=sr,
            stationary=True,
            prop_decrease=0.5  # Reduce effect to avoid excessive attenuation
        )

        # Apply low-pass filter (preserving more frequencies)
        nyquist = sr / 2
        cutoff = 7990 / nyquist  # Increase cutoff frequency
        b, a = sig.butter(5, cutoff, btype='low', analog=False)
        
        # Pass the reduced_noise to the filter
        filtered_signal = sig.filtfilt(b, a, reduced_noise)  # Corrected this line

        # Increase volume
        filtered_signal = filtered_signal * volume_increase_factor  # Increase volume by the specified factor

        # Normalize to restore volume levels
        filtered_signal = librosa.util.normalize(filtered_signal)

        return filtered_signal
    except Exception as e:
        st.error(f"Error in noise reduction: {e}")
        return audio_signal

@st.cache_resource
def load_tflite_model(model_path):
    """Load TFLite model with caching"""
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load TFLite model: {e}")
        return None

def process_audio(signal, sr, target_sr=16000):
    """Process audio to ensure consistent format"""
    try:
        # Resample if necessary
        if sr != target_sr:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
        
        # Convert to mono if stereo
        if len(signal.shape) > 1:
            signal = librosa.to_mono(signal)
        
        # Normalize audio
        signal = librosa.util.normalize(signal)
        
        # Apply initial noise reduction
        signal = apply_noise_reduction(signal, target_sr)
        
        return signal, target_sr
    except Exception as e:
        st.error(f"Error in audio processing: {e}")
        return signal, sr

def extract_features(signal, sr):
    """Extract mel spectrogram features"""
    try:
        # Apply pre-emphasis
        emphasized_signal = librosa.effects.preemphasis(signal)
        
        # Extract mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=emphasized_signal,
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            fmax=8000
        )
        
        # Convert to log scale
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return mel_spectrogram_db
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        return None

def enhance_audio(signal, sr):
    """Apply audio enhancement techniques"""
    try:
        # Apply spectral subtraction
        spec = librosa.stft(signal)
        mag = np.abs(spec)
        phase = np.angle(spec)
        
        # Estimate noise from the first few frames
        noise_estimate = np.mean(mag[:, :10], axis=1, keepdims=True)
        
        # Perform spectral subtraction
        mag_enhanced = np.maximum(mag - 1.5 * noise_estimate, 0)
        
        # Reconstruct signal
        enhanced = librosa.istft(mag_enhanced * np.exp(1j * phase))
        
        # Apply additional noise reduction
        enhanced = apply_noise_reduction(enhanced, sr)
        
        return enhanced
    except Exception as e:
        st.error(f"Error in audio enhancement: {e}")
        return signal

def run_tflite_inference(interpreter, input_features):
    """Run inference with proper error handling"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # Get input shape from model
        input_shape = input_details[0]['shape']
        
        # Reshape input to match model's expected shape
        input_data = input_features.reshape(input_shape).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        enhanced_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Reshape output back to audio signal shape
        enhanced_output = enhanced_output.reshape(-1)
        
        return enhanced_output
    except Exception as e:
        st.error(f"Inference error: {e}")
        return None

def save_uploaded_file(uploaded_file):
    """Save uploaded file and store in session state"""
    try:
        signal, sr = librosa.load(uploaded_file, sr=None)
        st.session_state.audio_data = signal
        st.session_state.sample_rate = sr
        
        # Save to temporary file for audio playback
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            sf.write(temp_audio.name, signal, sr)
            st.session_state.temp_file_path = temp_audio.name
        return True
    except Exception as e:
        st.error(f"Error saving audio file: {e}")
        return False

def record_audio(duration=5, samplerate=16000):
    """Record and save audio in session state"""
    try:
        st.write("Recording... Please speak into the microphone.")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        
        signal = audio.flatten()
        st.session_state.audio_data = signal
        st.session_state.sample_rate = samplerate
        
        # Save to temporary file for audio playback
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            sf.write(temp_audio.name, signal, samplerate)
            st.session_state.temp_file_path = temp_audio.name
        return True
    except Exception as e:
        st.error(f"Recording error: {e}")
        return False

# Application Interface
st.title("🎧 Noise Reduction System Using CNN")
st.subheader("Remove background noise and enhance your voice")

# Step 1: Input Speech Data
st.header("Step 1: Input Speech Data")
option = st.radio("Choose an input method:", ("Upload a File", "Record from Microphone"))

if option == "Upload a File":
    uploaded_file = st.file_uploader("Upload a Speech File (wav/mp3):", type=["wav", "mp3"])
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            st.write("Original Audio:")
            st.audio(st.session_state.temp_file_path
            , format="audio/wav")
            st.success("Audio file loaded successfully!")

elif option == "Record from Microphone":
    duration = st.slider("Recording Duration (seconds):", min_value=1, max_value=120, value=5)
    if st.button("Start Recording"):
        if record_audio(duration=duration):
            st.write("Original Recording:")
            st.audio(st.session_state.temp_file_path, format="audio/wav")
            st.success("Recording completed successfully!")

if st.session_state.audio_data is not None and st.session_state.sample_rate is not None:
    # Step 2: Process and Enhance Audio
    st.header("Step 2: Audio Processing")
    
    if st.button("Apply Noise Reduction"):
        # Process and enhance audio
        processed_signal, sr = process_audio(st.session_state.audio_data, st.session_state.sample_rate)
        enhanced_signal = enhance_audio(processed_signal, sr)
        
        # Save and play enhanced audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            output_path = temp_audio.name
            sf.write(output_path, enhanced_signal, sr)
            
            # Display spectrograms
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Spectrogram")
                fig_orig, ax_orig = plt.subplots(figsize=(10, 4))
                D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(st.session_state.audio_data)), ref=np.max)
                librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='hz')
                plt.colorbar(format='%+2.0f dB')
                st.pyplot(fig_orig)
            
            with col2:
                st.subheader("Enhanced Spectrogram")
                fig_enh, ax_enh = plt.subplots(figsize=(10, 4))
                D_enh = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_signal)), ref=np.max)
                librosa.display.specshow(D_enh, sr=sr, x_axis='time', y_axis='hz')
                plt.colorbar(format='%+2.0f dB')
                st.pyplot(fig_enh)
            
            # Play enhanced audio
            st.subheader("Clear with enhanced Audio")
            st.write("Clear Audio:")
            st.audio(output_path, format="audio/wav")
            
            # Provide download button
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Clear with Enhanced Audio",
                    data=file,
                    file_name="enhanced_audio.wav",
                    mime="audio/wav"
                )
            
            st.success("Clear audio enhancement completed successfully!")

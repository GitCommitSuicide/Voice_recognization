import streamlit as st
import os
import evaluate
import pandas as pd
import soundfile as sf
import numpy as np

st.set_page_config(page_title="Multi-Model Speech to Text Benchmark", layout="wide")

# Available Models grouped by type
MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "distil-whisper/distil-large-v3",
    "facebook/wav2vec2-base-960h",
    "facebook/hubert-large-ls960-ft",
    "patrickvonplaten/wavlm-libri-clean-100h-base-plus",
    "UsefulSensors/moonshine-base",
    "faster-whisper-tiny",
    "faster-whisper-base",
    "nvidia/canary-1b",
    "nvidia/parakeet-ctc-1.1b"
]

st.title("🏆 Speech to Text Multi-Model Benchmark & Ranker")
st.markdown("Compare top state-of-the-art models including **Whisper, Wav2Vec2, HuBERT, WavLM, Seamless M4T, Moonshine, Distil-Whisper, and Faster-Whisper**.")

st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect(
    "Select Models to Compare", 
    MODELS, 
    default=["openai/whisper-tiny", "facebook/wav2vec2-base-960h"]
)

@st.cache_resource(show_spinner=False)
def load_metrics():
    return evaluate.load("wer"), evaluate.load("cer")

wer_metric, cer_metric = load_metrics()

# Pipeline loader for HuggingFace models
@st.cache_resource(show_spinner=False)
def load_hf_pipeline(model_id):
    from transformers import pipeline
    
    return pipeline("automatic-speech-recognition", model=model_id, chunk_length_s=30, trust_remote_code=True)

st.write("Upload an audio file and provide the reference text. The app will transcribe the audio using all selected models, compute the errors, and rank them from best to worst.")

uploaded_file = st.file_uploader("Upload Audio File (wav, flac, etc.)", type=["wav", "mp3", "m4a", "flac"])
reference_text = st.text_area("Reference Text (Required for Ranking)", help="Ground truth transcript to calculate WER/CER and rank the models.")

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Benchmark Models"):
        if not selected_models:
            st.warning("Please select at least one model.")
        elif not reference_text.strip():
            st.warning("Please provide Reference Text to enable ranking.")
        else:
            temp_audio_path = "temp_audio_streamlit_upload.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            try:
                # Read for HF models
                audio_array, sampling_rate = sf.read(temp_audio_path)
                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=1)
                    
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Processing with {model_name}... ({i+1}/{len(selected_models)})")
                    try:
                        import time
                        start_time = time.time()
                        predicted_text = ""
                        
                        # Handle Faster Whisper separately as it doesn't use HF pipeline
                        if model_name.startswith("faster-whisper"):
                            from faster_whisper import WhisperModel
                            size = model_name.split("-")[-1]
                            # Load on CPU for compatibility, or change to cuda if available
                            model = WhisperModel(size, device="cpu", compute_type="int8")
                            segments, _ = model.transcribe(temp_audio_path, beam_size=5)
                            predicted_text = " ".join([segment.text for segment in segments]).strip()
                        
                        # Handle Nvidia NeMo models
                        elif model_name.startswith("nvidia/"):
                            import nemo.collections.asr as nemo_asr
                            if "canary" in model_name:
                                nemo_model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name)
                                res = nemo_model.transcribe([temp_audio_path])
                                predicted_text = res[0] if isinstance(res, list) else str(res)
                            else:
                                nemo_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)
                                res = nemo_model.transcribe([temp_audio_path])
                                predicted_text = res[0] if isinstance(res, list) else str(res)
                                
                            if isinstance(predicted_text, list):
                                predicted_text = predicted_text[0]
                                
                        # Handle standard HuggingFace models
                        else:
                            asr = load_hf_pipeline(model_name)
                            prediction = asr({"array": audio_array, "sampling_rate": sampling_rate})
                            predicted_text = prediction["text"].strip()
                            
                        end_time = time.time()
                        latency = round(end_time - start_time, 2)
                        
                        orig_lower = reference_text.lower().strip()
                        pred_lower = predicted_text.lower().strip()
                        
                        wer = wer_metric.compute(predictions=[pred_lower], references=[orig_lower])
                        cer = cer_metric.compute(predictions=[pred_lower], references=[orig_lower])
                        
                        results.append({
                            "Model": model_name,
                            "WER": round(wer, 4),
                            "CER": round(cer, 4),
                            "Latency (s)": latency,
                            "Transcription": predicted_text
                        })
                    except Exception as e:
                        st.error(f"Error with {model_name}: {e}")
                        
                    progress_bar.progress((i + 1) / len(selected_models))
                    
                status_text.text("Benchmarking Complete!")
                
                if results:
                    df = pd.DataFrame(results)
                    # Rank by WER (lowest is best)
                    df = df.sort_values(by="WER", ascending=True).reset_index(drop=True)
                    # Add ranking column
                    df.index += 1
                    df.reset_index(inplace=True)
                    df.rename(columns={'index': 'Rank'}, inplace=True)
                    
                    st.subheader("🏆 Model Rankings (Best to Worst)")
                    
                    # Display DataFrame with styling
                    st.dataframe(
                        df.style.highlight_min(subset=['WER', 'CER', 'Latency (s)'], color='rgba(144, 238, 144, 0.5)')
                              .format({'WER': "{:.2%}", 'CER': "{:.2%}", 'Latency (s)': "{:.2f} s"}),
                        width='stretch'
                    )
                    
                    st.subheader("📝 Detailed Transcriptions")
                    # Create a tab for each model in ranked order
                    tabs = st.tabs([f"#{row['Rank']} {row['Model']}" for _, row in df.iterrows()])
                    
                    for tab, (_, row) in zip(tabs, df.iterrows()):
                        with tab:
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Word Error Rate (WER)", f"{row['WER']:.2%}")
                            col2.metric("Character Error Rate (CER)", f"{row['CER']:.2%}")
                            col3.metric("Latency", f"{row['Latency (s)']} s")
                            st.info(row['Transcription'])
                            
            except Exception as e:
                st.error(f"Error processing audio: {e}")
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

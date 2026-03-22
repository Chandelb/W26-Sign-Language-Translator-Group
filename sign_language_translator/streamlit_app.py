"""
ASL Sign Recognition — Streamlit Starter Code

IMPORTANT: Before clicking "Record Sign", you must first click the
"START" button to activate the video stream!

Usage:
    pip install streamlit streamlit-webrtc av torch pandas mediapipe opencv-python-headless
    streamlit run streamlit_app.py
"""

import time
import threading
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import av
import streamlit as st
from pathlib import Path
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from cleaning_file import clean
from duplicates_and_suffixes import handle_duplicates
from llm_inference import infer

from asl_citizen_processor import Extractor, FEATURE_DIM
from lstm_model import Video_LSTM


# st.set_page_config must be the first Streamlit call in the script.
# layout="wide" makes the app use the full browser width.
st.set_page_config(page_title="ASL Sign Recognition", page_icon="🤟", layout="wide")


# TODO: update HIDDEN_SIZE and N_LAYERS to match your trained model.
# If you get a "size mismatch" error on startup, these are wrong.
PROCESSED_DIR = "asl_citizen_processed"
MODEL_PATH    = "saved_models/asl_citizen_fc_model.pth"
HIDDEN_SIZE   = 256   # TODO: change to match your training config
N_LAYERS      = 4     # TODO: change to match your training config
DROPOUT       = 0.5
TOP_K         = 3     # how many top predictions to show

COUNTDOWN_SEC = 3     # seconds of countdown before recording starts
RECORD_SEC    = 3     # seconds of recording per sign

# the app moves through these four states in order
IDLE      = "idle"       # waiting for the user to press the button
COUNTDOWN = "countdown"  # counting down before recording
RECORDING = "recording"  # actively capturing and extracting landmarks
DONE      = "done"       # prediction is ready

# pairs of landmark indices to connect when drawing the hand skeleton.
# MediaPipe gives us 21 landmarks per hand numbered 0-20.
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# @st.cache_resource runs this function only once and reuses the result across
# all reruns. Without it, the model would reload from disk every time the user
# clicks a button, which takes several seconds each time.
@st.cache_resource
def load_resources():
    processed_path = Path(PROCESSED_DIR)

    # the label map maps integer class indices to gloss strings ("HELLO", "THANK YOU", etc.)
    label_map      = pd.read_csv(processed_path / "label_map.csv")
    label_to_gloss = dict(zip(label_map["label"].astype(int), label_map["gloss"]))
    num_classes    = len(label_to_gloss)

    # feature_dim is the size of each frame's landmark tensor (84 for two hands)
    config_path = processed_path / "config.csv"
    feature_dim = (int(pd.read_csv(config_path).iloc[0]["feature_dim"])
                   if config_path.exists() else FEATURE_DIM)

    # pick the best available device: MPS on Apple Silicon, CUDA on NVIDIA, else CPU
    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available()           else
              torch.device("cpu"))

    # TODO: make sure these hyperparameters exactly match your trained model.
    # The architecture here must match what was saved to disk or you'll get a size mismatch.
    model = Video_LSTM(
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        num_layers=N_LAYERS,
        num_classes=num_classes,
        input_size=feature_dim,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()  # puts the model in inference mode (disables dropout etc.)
    model = model.to(device)

    return model, label_to_gloss, device


def run_predict(model, frames, label_to_gloss, device):
    """
    Takes a list of landmark tensors (one per frame) and returns top-K predictions.
    Each prediction is a (gloss_string, probability) tuple.
    """
    if len(frames) < 5:
        # the LSTM needs a minimum number of frames to make a meaningful prediction
        return [("(too short — try again)", 0.0)]

    # frames is a list of 1D tensors of shape (feature_dim,).
    # torch.stack turns it into (T, feature_dim), unsqueeze(0) adds the batch dim -> (1, T, feature_dim).
    video = torch.stack(frames).unsqueeze(0).to(device)

    with torch.no_grad():
        # torch.no_grad() skips building a computation graph since we're doing inference not training.
        # softmax converts raw logits to probabilities that sum to 1.
        probs = F.softmax(model(video), dim=1)[0]

    k = min(TOP_K, len(label_to_gloss))
    top_probs, top_idx = probs.topk(k)
    return [(label_to_gloss[i.item()], p.item()) for i, p in zip(top_idx, top_probs)]


def draw_landmarks(frame, result):
    """Draws the hand skeleton directly onto a frame (modifies it in place)."""
    h, w = frame.shape[:2]
    for hand_lms in result.hand_landmarks:
        # landmark x/y are normalized 0-1, multiply by width/height to get pixel coords
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)


try:
    model, label_to_gloss, device = load_resources()
except Exception as e:
    st.error(f"Could not load model/data: {e}")
    st.info("Make sure PROCESSED_DIR and MODEL_PATH at the top of the file are correct.")
    st.stop()


# shared state is a plain dict stored via @st.cache_resource so it's truly global
# both the Streamlit UI thread and webrtc's recv() thread can read and write it.
# Normal st.session_state won't work here because recv() runs on a background thread
# that Streamlit doesn't control.
# We use a threading.Lock any time we read or write this dict to prevent race conditions
# (two threads writing at the same time would corrupt the data).
@st.cache_resource
def get_shared_state():
    return {
        "lock":            threading.Lock(),
        "app_state":       IDLE,
        "state_start":     0.0,
        "recorded_frames": [],
        "results":         [],
        "history":         deque(maxlen=8),
    }

shared = get_shared_state()


# ASLProcessor is passed to webrtc_streamer as a class (not an instance).
# streamlit-webrtc instantiates it when the stream starts and calls recv()
# for every incoming video frame in a background thread.
# All mutable state lives in `shared` above so the UI can also read/write it.
class ASLProcessor:
    def __init__(self):
        # the Extractor wraps MediaPipe and handles landmark extraction frame by frame.
        # we call __enter__ manually because we're not using a `with` statement.
        self.extractor = Extractor().__enter__()

    def recv(self, frame):
        # convert the incoming webrtc frame to a numpy BGR image
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror so it feels like looking in a mirror

        # always read shared state under the lock to avoid race conditions
        with shared["lock"]:
            state   = shared["app_state"]
            elapsed = time.time() - shared["state_start"]

        if state == COUNTDOWN:
            if elapsed >= COUNTDOWN_SEC:
                # countdown finished switch to recording
                with shared["lock"]:
                    shared["app_state"]       = RECORDING
                    shared["state_start"]     = time.time()
                    shared["recorded_frames"] = []
                    elapsed                   = 0.0
                    state                     = RECORDING
                self.extractor.reset_timestamp()
            else:
                # draw the countdown number in big text on the frame
                remaining = int(COUNTDOWN_SEC - elapsed) + 1
                h, w = img.shape[:2]
                cv2.putText(img, str(remaining),
                            (w // 2 - 40, h // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 8)

        if state == RECORDING:
            # extract_with_result returns a landmark tensor t (or None if no hand detected)
            # and a MediaPipe result object used for drawing landmarks
            t, mp_result = self.extractor.extract_with_result(img)
            if t is not None:
                with shared["lock"]:
                    shared["recorded_frames"].append(t.cpu())
            if mp_result is not None:
                draw_landmarks(img, mp_result)

            # draw a progress bar at the bottom of the video frame
            h, w     = img.shape[:2]
            progress = min(1.0, elapsed / RECORD_SEC)
            cv2.rectangle(img, (0, h - 10), (int(progress * w), h),
                          (0, 80, 255), -1)

            if elapsed >= RECORD_SEC:
                with shared["lock"]:
                    frames_copy         = list(shared["recorded_frames"])
                    shared["app_state"] = DONE

                # TODO: call run_predict() with the right arguments
                results = run_predict(model, frames_copy, label_to_gloss, device)
                with shared["lock"]:
                    shared["results"] = results
                    shared["history"].append(results[0][0])

        # when idle or done, still run the extractor so landmarks are drawn live —
        # this gives the user visual feedback that their hands are being detected
        if state in (IDLE, DONE):
            _, mp_result = self.extractor.extract_with_result(img)
            if mp_result is not None:
                draw_landmarks(img, mp_result)

        # return the annotated frame back to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("🤟 ASL Sign Recognition")

col_cam, col_panel = st.columns([3, 2], gap="large")

with col_cam:
    # webrtc_streamer handles the camera in a background thread.
    # IMPORTANT: the user must click START in this widget before clicking Record Sign.
    # async_processing=False means recv() is called synchronously — simpler and more
    # reliable than async on most machines.
    webrtc_streamer(
        key="asl",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=ASLProcessor,
        async_processing=False,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶  Record Sign", type="primary", use_container_width=True):
            # clicking the button updates shared state — recv() picks it up on the next frame
            with shared["lock"]:
                shared["app_state"]       = COUNTDOWN
                shared["state_start"]     = time.time()
                shared["recorded_frames"] = []
                shared["results"]         = []
    with col2:
        if st.button("✕  Cancel", use_container_width=True):
            with shared["lock"]:
                shared["app_state"]       = IDLE
                shared["recorded_frames"] = []

    if st.button("↺  Clear History", use_container_width=True):
        with shared["lock"]:
            shared["app_state"]       = IDLE
            shared["recorded_frames"] = []
            shared["results"]         = []
            shared["history"]         = deque(maxlen=8)

with col_panel:
    st.subheader("Predictions")
    prediction_placeholder = st.empty()

    st.subheader("History")
    history_placeholder = st.empty()

    # TODO (extension): add anything else you want in this panel!
    # ideas: confidence threshold slider, device info, sign reference images,
    # a sentence builder that strings history signs together...


# read the latest results from shared state and render them.
# this runs on every Streamlit rerun (i.e. every button click).
# the predictions won't auto-refresh between clicks — that's fine for now,
# since a new prediction only appears after a full recording cycle anyway.
with shared["lock"]:
    results = list(shared["results"])
    history = list(shared["history"])
    # TODO (extension): -history 
# TODO : cleaning_file.py: history -> cleaning file, what this cleaning file should incorporate putting consecutive letters andd putting it into a word
# TODO : add a hotkey to detect spacebar as "\n" in the history
# TODO : add an end gesture to detect spacebar as "\n" in the history
# TODO : duplicates_and_suffixes.py: add point in processing pipeline that uses the sign-language-translator Python library to clean up the string (remove duplicates, handle common ASL suffixes), and handle spaces.
# TODO : llm_inference.py Inference: Pass that normalized string into a small, fine-tuned LLM (like a T5-base model hosted on Hugging Face) or use an LLM prompt like: "Translate the following ASL Glosses into a natural English sentence: [YOUR_STRING_HERE]"

# TODO (REACH) : get time from last gesture, incorporate into all other logic
    clean(list)
    handle_duplicates(list)
    infer(list)

    



# TODO (extension): make show_predictions and show_history look however you want!
# you can use st.metric(), progress bars, custom CSS.
if not results:
    prediction_placeholder.markdown("*No prediction yet — click START then Record Sign.*")
else:
    lines = []
    for i, (gloss, prob) in enumerate(results):
        prefix = "→" if i == 0 else "  "
        lines.append(f"{prefix} **{gloss}** — {prob*100:.1f}%")
    prediction_placeholder.markdown("\n\n".join(lines))

if not history:
    history_placeholder.markdown("*Your recent signs will appear here.*")
else:
    history_placeholder.markdown("  ·  ".join(reversed(history)))

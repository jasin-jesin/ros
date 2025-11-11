
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from faster_whisper import WhisperModel
from openai import OpenAI
import numpy as np
import subprocess
import os
from datetime import datetime

PIPER_PATH = "/home/pi/.local/bin/piper"  # Adjust if needed
VOICE_PATH = "/home/pi/Downloads/en_GB-southern_english_female-low.onnx"
TMP_WAV = "/tmp/speech.wav"

model = WhisperModel("base.en", compute_type="int8")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class VoiceAISubscriberNode(Node):
    def __init__(self):
        super().__init__('voice_ai_subscriber_node')
        self.subscription = self.create_subscription(String, 'mic_data', self.listener_callback, 10)

    def speak_piper(self, text):
        cmd = [
            PIPER_PATH,
            "--model", VOICE_PATH,
            "--text", text,
            "--output-file", TMP_WAV
        ]
        subprocess.run(cmd, check=True)
        subprocess.run(["aplay", TMP_WAV], check=True)

    def stream_gpt(self, prompt):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Neo, a concise robot assistant. Keep answers under 3 sentences."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            max_tokens=80
        )
        full_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                piece = delta.content
                full_reply += piece
                print(piece, end="", flush=True)
        print()
        self.speak_piper(full_reply)

    def listener_callback(self, msg):
        audio_data = bytes.fromhex(msg.data)
        audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0
        segments, info = model.transcribe(audio_np, beam_size=1, vad_filter=True)
        text = " ".join([seg.text for seg in segments]).strip().lower()
        print("🗣️", text)

        if not text:
            return

        if "time" in text:
            self.speak_piper(datetime.now().strftime("The time is %I:%M %p."))
        elif "weather" in text:
            self.speak_piper("Sorry, weather not available offline.")
        elif "youtube" in text:
            os.system("xdg-open https://www.youtube.com")
            self.speak_piper("Opening YouTube.")
        else:
            self.stream_gpt(text)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceAISubscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

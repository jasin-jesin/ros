
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pyaudio
import collections
import webrtcvad

class MicPublisherNode(Node):
    def __init__(self):
        super().__init__('mic_publisher_node')
        self.publisher_ = self.create_publisher(String, 'mic_data', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.vad = webrtcvad.Vad(2)
        self.pa = pyaudio.PyAudio()
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.frames_per_buffer = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.stream = self.pa.open(
            rate=self.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.frames_per_buffer
        )
        self.ring_buffer = collections.deque(maxlen=int(300 / self.frame_duration_ms))
        self.triggered = False
        self.voiced_frames = []

    def timer_callback(self):
        frame = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
        is_speech = self.vad.is_speech(frame, self.sample_rate)
        if not self.triggered:
            self.ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in self.ring_buffer if speech])
            if num_voiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = True
                self.voiced_frames.extend([f for f, s in self.ring_buffer])
                self.ring_buffer.clear()
        else:
            self.voiced_frames.append(frame)
            self.ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
            if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                audio_data = b"".join(self.voiced_frames)
                self.voiced_frames = []
                self.triggered = False
                # Publish raw audio data as hex string
                msg = String()
                msg.data = audio_data.hex()
                self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MicPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

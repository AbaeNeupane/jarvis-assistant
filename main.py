# --- IMPORTANT: This main.py goes in the ROOT directory ---

# Import from our custom source and UI packages
from src.wake_word_module import WakeWordDetector
from src import stt_module
from src import llm_module
from src import tts_module
from ui.app import update_status, add_message, run_ui

# Import utilities
import time
import threading

class JarvisAssistant:
    def __init__(self):
        """
        Initializes the main Jarvis Assistant application.
        """
        self.state = "LISTENING"
        self.lock = threading.Lock()

        jarvis_prompt = (
            "You are Jarvis, a helpful, witty, and slightly sarcastic AI assistant. "
            "You are running locally on the user's computer. Keep your responses concise."
        )
        self.llm_client = llm_module.LLMClient(system_prompt=jarvis_prompt)
        self.wake_word_detector = WakeWordDetector(on_activation=self.on_wake_word_activated)

        print("Jarvis Assistant initialized.")


    def on_wake_word_activated(self):
        """
        Callback function that runs when "Hey Jarvis" is detected.
        """
        with self.lock:
            if self.state == "PROCESSING":
                print("Already processing, ignoring new activation.")
                return
            self.state = "PROCESSING"
            update_status("Listening...")

        try:
            update_status("Recording...")
            transcribed_text = stt_module.record_and_transcribe()
            if not transcribed_text or len(transcribed_text.strip()) < 2:
                print("No clear text transcribed.")
                tts_module.speak("I'm sorry, I didn't catch that.")
                return

            add_message("You", transcribed_text)
            update_status("Thinking...")
            llm_response = self.llm_client.get_response(transcribed_text)
            if not llm_response:
                print("No response from LLM.")
                return
            
            add_message("Jarvis", llm_response)
            update_status("Speaking...")
            tts_module.speak(llm_response)

        except Exception as e:
            print(f"An error occurred in the action loop: {e}")
        finally:
            self.state = "LISTENING"
            update_status("Listening...")

    def run(self):
        """
        Starts the UI and the wake word detector.
        """
        try:
            ui_thread = threading.Thread(target=run_ui, daemon=True)
            ui_thread.start()
            time.sleep(2)
            update_status("Listening...")
            add_message("Jarvis", "Hello. I am online and listening.")
            self.wake_word_detector.start()
            
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("Shutting down Jarvis.")
            self.wake_word_detector.stop()
        except FileNotFoundError as e:
            print(f"\n[FATAL ERROR] A required file was not found: {e}")
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")


if __name__ == '__main__':
    assistant = JarvisAssistant()
    assistant.run()

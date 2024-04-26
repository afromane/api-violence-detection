import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from datetime import datetime
import time
from collections import deque
import tensorflow as tf
from django.conf import settings
import keras

class DectectViolenceAPI:
    IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
    SEQUENCE_LENGTH = 24
    CLASSES_LIST = ["NonViolence", "Violence"]

    def __init__(self, save_target='test_video', violence_threshold=10, detection_interval=60, detected_frames_dir='detected_frames'):
        self.model_list = [
            settings.BASE_DIR+"/static/model/seq24/model_0_300_f600.h5",
            settings.BASE_DIR+"/static/model/seq24/model_300_600_f600.h5",
            settings.BASE_DIR+"/static/model/seq24/model_600_900_f506.h5",
            settings.BASE_DIR+"/static/model/seq24/1.h5",
        ]
        self.model_list = [keras.models.load_model(model_path) for model_path in self.model_list]
        self.detected_frames_dir = detected_frames_dir
        self.violence_threshold = violence_threshold
        self.detection_interval = detection_interval

        os.makedirs(self.detected_frames_dir, exist_ok=True)

        current_time = datetime.now().time()
        formatted_time = current_time.strftime("%H-%M-%S")
        self.output_file_path = f'{save_target}/Output_{formatted_time}.mp4'

    def predict_frames_parallel(self, video_file_path):
        video_reader = cv2.VideoCapture(video_file_path)
        total_video_duration = video_reader.get(cv2.CAP_PROP_FRAME_COUNT) / video_reader.get(cv2.CAP_PROP_FPS)

        violence_count = 0
        non_violence_count = 0
        frames_queue = deque(maxlen=self.SEQUENCE_LENGTH)
        detected_frames = []
        detection_times = []
        def predict_with_model(model, frames_queue):
            try:
                return model.predict(np.expand_dims(frames_queue, axis=0))[0]
            except Exception as e:
                print(f"Error predicting with model: {model}, {e}")
                return None

        while video_reader.isOpened():
            ok, frame = video_reader.read()

            if not ok:
                break

            resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_queue.append(normalized_frame)
            all_predicted_probabilities = []

            if len(frames_queue) == self.SEQUENCE_LENGTH:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(predict_with_model, model, frames_queue) for model in self.model_list]

                    for future in concurrent.futures.as_completed(futures):
                        predicted_labels_probabilities = future.result()
                        if predicted_labels_probabilities is not None:
                            all_predicted_probabilities.append(predicted_labels_probabilities)

                average_predicted_probabilities = np.mean(all_predicted_probabilities, axis=0)
                predicted_label_index = np.argmax(average_predicted_probabilities)

                predicted_label = self.CLASSES_LIST[predicted_label_index]

                if predicted_label == "Violence":
                    violence_count += 1
                    detected_frames.append(frame)
                else:
                    non_violence_count += 1

            elapsed_time = video_reader.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if elapsed_time >= self.detection_interval:
                break

        video_reader.release()

        violence_percentage = (violence_count / (violence_count + non_violence_count)) * 100
        non_violence_percentage = 100 - violence_percentage

        if violence_count >= self.violence_threshold:
            save_directory = self.save_detected_frames(detected_frames, elapsed_time)
            detection_times.append(elapsed_time)  # Convert to seconds

            return violence_percentage, non_violence_percentage, save_directory,detection_times
        else:
            return violence_percentage, non_violence_percentage, None,None
    
    def save_detected_frames(self, frames, detection_start_time):
        current_date = time.strftime("%Y-%m-%d", time.localtime(detection_start_time))
        interval_start = time.strftime("%H-%M-%S", time.localtime(detection_start_time))
        interval_end = time.strftime("%H-%M-%S", time.localtime(detection_start_time + self.detection_interval))
        interval_folder_name = f'{current_date}/{interval_start}:{interval_end}'

        save_directory = os.path.join(self.detected_frames_dir, interval_folder_name)

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        for idx, frame in enumerate(frames):
            cv2.imwrite(os.path.join(save_directory, f'detected_frame_{idx}.jpg'), frame)

        return save_directory


    def predict_images(self, images_list):
        
      frames_queue = deque(maxlen=self.SEQUENCE_LENGTH)
      predicted_class_name = ''

      def predict_with_model(model, frames_queue):
          try:
              return model.predict(np.expand_dims(frames_queue, axis=0))[0]
          except Exception as e:
              print(f"Error predicting with model: {model}, {e}")
              return None

      all_predicted_probabilities = []

      for frame in images_list:

          resized_frame = cv2.resize(frame, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
          normalized_frame = resized_frame / 255
          frames_queue.append(normalized_frame)

          if len(frames_queue) == self.SEQUENCE_LENGTH:
              with concurrent.futures.ThreadPoolExecutor() as executor:
                  futures = [executor.submit(predict_with_model, model, frames_queue) for model in self.model_list]

                  for future in concurrent.futures.as_completed(futures):
                      predicted_labels_probabilities = future.result()
                      if predicted_labels_probabilities is not None:
                          all_predicted_probabilities.append(predicted_labels_probabilities)

              average_predicted_probabilities = np.mean(all_predicted_probabilities, axis=0)
              predicted_label = np.argmax(average_predicted_probabilities)
              predicted_class_name = self.CLASSES_LIST[predicted_label]

      return predicted_class_name,average_predicted_probabilities

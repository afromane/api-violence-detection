import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from datetime import datetime

from collections import deque
import tensorflow as tf
from django.conf import settings
import keras

class DectectViolenceAPI:
    IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
    SEQUENCE_LENGTH = 24
    CLASSES_LIST = ["NonViolence", "Violence"]

    def __init__(self, save_target='test_video'):

   
        self.model_list = [
            #settings.STATIC_ROOT+ "model/model_0_500.h5"
            settings.BASE_DIR+"/static/model/seq24/model_0_300_f600.h5",
            settings.BASE_DIR+"/static/model/seq24/model_300_600_f600.h5",
            settings.BASE_DIR+"/static/model/seq24/model_600_900_f600.h5",
            # settings.BASE_DIR+"/static/model/seq24/model_900_1153_f506.h5",
        ]

        self.model_list = [keras.models.load_model(model_path) for model_path in self.model_list]

        os.makedirs(save_target, exist_ok=True)

        current_time = datetime.now().time()
        formatted_time = current_time.strftime("%H-%M-%S")
        self.output_file_path = f'{save_target}/Output_{formatted_time}.mp4'
    
    def predict_frames_parallel(self, video_file_path):
        violence_prediction_timestamps =[]
        video_reader = cv2.VideoCapture(video_file_path)
        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(self.output_file_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                       video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
        frames_queue = deque(maxlen=self.SEQUENCE_LENGTH)
        predicted_class_name = ''

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
                predicted_label = np.argmax(average_predicted_probabilities)
                predicted_class_name = self.CLASSES_LIST[predicted_label]

            if predicted_class_name == "Violence":
                cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
                violence_prediction_timestamps.append(video_reader.get(cv2.CAP_PROP_POS_MSEC))

            """  else:
                cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12) """

            video_writer.write(frame)

        video_reader.release()
        video_writer.release()
        return violence_prediction_timestamps
   
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


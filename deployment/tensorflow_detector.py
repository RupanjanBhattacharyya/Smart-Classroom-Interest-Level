import time
import cv2
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications.mobilenet import preprocess_input
from collections import Counter
import matplotlib.pyplot as plt

emotion_classes = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Afraid', 'Disgusted', 'Angry', 'Contemptuous']
interest_classes = ['Not Interested', 'Interested']

class TensorflowDetector(object):
    def __init__(self, PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS):
        self.emotion_history = []
        self.interest_history = []
        self.valence_history = []
        self.arousal_history = []

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess_1 = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

        self.classification_graph = tf.Graph()
        with self.classification_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CLASS, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.classification_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess_2 = tf.compat.v1.Session(graph=self.classification_graph, config=config)

        self.regression_graph = tf.Graph()
        with self.regression_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_REGRESS, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.regression_graph.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess_3 = tf.compat.v1.Session(graph=self.regression_graph, config=config)

    def map_emotion_to_interest(self, emotion):
        interested_emotions = ['Happy', 'Neutral', 'Surprised']
        return 'Interested' if emotion in interested_emotions else 'Not Interested'

    def generate_report(self):
        if not self.emotion_history:
            return "No data collected for report generation"

        emotion_counts = Counter(self.emotion_history)
        interest_counts = Counter(self.interest_history)
        
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        dominant_emotion_score = emotion_counts[dominant_emotion] / len(self.emotion_history)
        
        dominant_interest = interest_counts.most_common(1)[0][0]
        dominant_interest_score = interest_counts[dominant_interest] / len(self.interest_history)
        
        attention_index = interest_counts['Interested'] / len(self.interest_history) if self.interest_history else 0
        
        avg_valence = np.mean(self.valence_history) if self.valence_history else 0
        avg_arousal = np.mean(self.arousal_history) if self.arousal_history else 0
        emotion_index = (avg_valence + avg_arousal) / 2
        
        # Calculate interest level score similar to emotion index
        interest_level_score = attention_index * ((avg_valence + 1) / 2)  # Normalize valence to 0-1 range

        report = f"""
=== Interest Level Report ===
Dominant Emotion: {dominant_emotion} ({dominant_emotion_score:.2%})
Dominant Interest Level: {dominant_interest} ({dominant_interest_score:.2%})
Attention Index: {attention_index:.2%}
Emotion Index: {emotion_index:.3f}
Interest Level Score: {interest_level_score:.3f}

Detailed Metrics:
- Total Observations: {len(self.emotion_history)}
- Emotion Distribution: {dict(emotion_counts)}
- Interest Distribution: {dict(interest_counts)}
- Average Valence: {avg_valence:.3f}
- Average Arousal: {avg_arousal:.3f}
========================
"""
        return report

    def plot_detector_results(self):
        """
        Plot the results from a TensorflowDetector object
        
        Args:
            detector: TensorflowDetector object with history data
        
        Returns:
            matplotlib figure object
        """
        if not self.emotion_history:
            print("No data available for plotting")
            return None

        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Emotion Distribution
        plt.subplot(2, 2, 1)
        emotion_counts = Counter(self.emotion_history)
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        plt.bar(emotions, counts)
        plt.title('Emotion Distribution')
        plt.xticks(rotation=45)
        plt.ylabel('Count')

        # Plot 2: Interest Level Distribution
        plt.subplot(2, 2, 2)
        interest_counts = Counter(self.interest_history)
        plt.pie(interest_counts.values(), 
                labels=interest_counts.keys(),
                autopct='%1.1f%%',
                colors=['lightcoral', 'lightgreen'])
        plt.title('Interest Level Distribution')

        # Plot 3: Valence-Arousal Scatter Plot
        plt.subplot(2, 2, 3)
        plt.scatter(self.valence_history, self.arousal_history, alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Valence-Arousal Distribution')
        plt.xlabel('Valence')
        plt.ylabel('Arousal')

        # Plot 4: Time Series of Interest Level
        plt.subplot(2, 2, 4)
        interest_binary = [1 if x == 'Interested' else 0 for x in self.interest_history]
        plt.plot(range(len(interest_binary)), interest_binary, 'g-')
        plt.title('Interest Level Over Time')
        plt.xlabel('Observation')
        plt.ylabel('Interest Level (0=Not Interested, 1=Interested)')
        plt.grid(True)

        plt.tight_layout()
        return fig

    def run(self, image):
        [h, w] = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess_1.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        print('Detection time: {}'.format(round(time.time() - start_time, 8)))
        print('--------------------------------------')

        classification_input = self.classification_graph.get_tensor_by_name('input_1:0')
        classification_output = self.classification_graph.get_tensor_by_name('dense_2/Softmax:0')
        regression_input = self.regression_graph.get_tensor_by_name('input_1:0')
        regression_output = self.regression_graph.get_tensor_by_name('dense_2/BiasAdd:0')

        images_for_prediction = []
        for i in range(min(20, np.squeeze(boxes).shape[0])):
            if scores is None or np.squeeze(scores)[i] > 0.7:
                ymin, xmin, ymax, xmax = np.squeeze(boxes)[i]
                image_pred = image[max(int(h * ymin) - 20, 0):min(int(h * ymax) + 20, image.shape[:2][0]),
                             max(int(w * xmin) - 20, 0):min(int(w * xmax) + 20, image.shape[:2][1]), :]
                image_pred = Image.fromarray(image_pred).resize((224, 224))
                image_pred = keras.preprocessing.image.img_to_array(image_pred)
                image_pred = preprocess_input(image_pred)
                images_for_prediction.append(image_pred)

        emotions_detected = []
        interest_levels = []
        if len(images_for_prediction) > 0:
            start_time = time.time()
            prediction = self.sess_2.run(classification_output,
                                       feed_dict={classification_input: images_for_prediction})
            print('Classification time: {}'.format(round(time.time() - start_time, 8)))
            
            for row in prediction:
                pred = np.argmax(row)
                emotion = emotion_classes[pred]
                interest = self.map_emotion_to_interest(emotion)
                emotion_score = round(row[pred], 2)
                interest_score = emotion_score * (1 if interest == 'Interested' else 0.5)  # Weight based on interest level
                
                print(f"{emotion} ({interest}) - Emotion Score: {emotion_score:.2f}, Interest Score: {interest_score:.2f}")
                
                self.emotion_history.append(emotion)
                self.interest_history.append(interest)
                emotions_detected.append(emotion)
                interest_levels.append(interest)
            
            print('--------------------------------------')
            
            start_time = time.time()
            prediction = self.sess_3.run(regression_output,
                                       feed_dict={regression_input: images_for_prediction})
            print('Regression time: {}'.format(round(time.time() - start_time, 8)))
            
            for row in prediction:
                valence = row[0]
                arousal = row[1]
                print(f'Valence: {round(valence, 5)} Arousal: {round(arousal, 5)}')
                
                self.valence_history.append(valence)
                self.arousal_history.append(arousal)

            print('\n')

        return boxes, scores, classes, num_detections, interest_levels
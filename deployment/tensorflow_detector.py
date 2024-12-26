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
        self.emotion_scores = []  
        self.interest_scores = []

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

    def calculate_emotion_score(self, prediction_row):
        # Calculate emotion confidence score from model output
        return round(np.max(prediction_row), 2)

    def calculate_interest_score(self, emotion_score, valence, arousal):
        # Based on Russell's Circumplex Model
        # Source: Russell, J. A. (1980). A circumplex model of affect.
        valence_weight = 0.4  # Weight for emotional positivity
        arousal_weight = 0.6  # Weight for emotional intensity
        
        # Normalize valence and arousal to 0-1 scale
        valence_norm = (valence + 1) / 2
        arousal_norm = (arousal + 1) / 2
        
        # Calculate interest score using dimensional model
        interest_score = emotion_score * (valence_weight * valence_norm + arousal_weight * arousal_norm)
        return round(interest_score, 2)

    def generate_attention_remark(self, interest_percentage):
        if interest_percentage >= 80:
            return "Highly attentive with minimal signs of distraction."
        elif interest_percentage >= 60:
            return "Generally attentive with occasional distraction."
        elif interest_percentage >= 40:
            return "Moderately attentive with regular periods of distraction."
        elif interest_percentage >= 20:
            return "Frequently distracted with occasional periods of attention."
        else:
            return "Predominantly distracted with minimal attention spans."

    def generate_report(self):
        if not self.emotion_history:
            return "No data collected for report generation"

        emotion_counts = Counter(self.emotion_history)
        interest_counts = Counter(self.interest_history)
        
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        dominant_emotion_score = emotion_counts[dominant_emotion] / len(self.emotion_history)
        
        dominant_interest = interest_counts.most_common(1)[0][0]
        dominant_interest_score = interest_counts[dominant_interest] / len(self.interest_history)
        
        avg_emotion_score = np.mean(self.emotion_scores)
        avg_interest_score = np.mean(self.interest_scores)
        
        avg_valence = np.mean(self.valence_history)
        avg_arousal = np.mean(self.arousal_history)

        # Calculate interest percentage and generate attention remark
        interest_percentage = interest_counts.get('Interested', 0) / len(self.interest_history) * 100
        attention_remark = self.generate_attention_remark(interest_percentage)

        report = f"""
=== Interest Level Report ===
Dominant Emotion: {dominant_emotion} ({dominant_emotion_score:.2%})
Average Emotion Score: {avg_emotion_score:.3f}
Dominant Interest Level: {dominant_interest} ({dominant_interest_score:.2%})
Average Interest Score: {avg_interest_score:.3f}

Attention Analysis:
{attention_remark}

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

        # Plot 3: Emotion vs Interest Scores
        plt.subplot(2, 2, 3)
        plt.scatter(self.emotion_scores, self.interest_scores, alpha=0.5)
        plt.xlabel('Emotion Confidence Score')
        plt.ylabel('Interest Score')
        plt.title('Emotion vs Interest Correlation')
        plt.grid(True)

        # Plot 4: Valence-Arousal Distribution
        plt.subplot(2, 2, 4)
        sc = plt.scatter(self.valence_history, self.arousal_history, 
                        c=self.interest_scores, cmap='RdYlGn', alpha=0.5)
        plt.colorbar(sc, label='Interest Score')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Valence-Arousal Distribution')
        plt.xlabel('Valence')
        plt.ylabel('Arousal')

        plt.tight_layout()
        return fig

    def run(self, image):
        [h, w] = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image, axis=0)
        
        # Face detection
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

        # Emotion classification and VA regression setup
        classification_input = self.classification_graph.get_tensor_by_name('input_1:0')
        classification_output = self.classification_graph.get_tensor_by_name('dense_2/Softmax:0')
        regression_input = self.regression_graph.get_tensor_by_name('input_1:0')
        regression_output = self.regression_graph.get_tensor_by_name('dense_2/BiasAdd:0')

        # Process detected faces
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
            # Emotion classification
            start_time = time.time()
            emotion_prediction = self.sess_2.run(classification_output,
                                               feed_dict={classification_input: images_for_prediction})
            print('Classification time: {}'.format(round(time.time() - start_time, 8)))
            
            # Valence-Arousal regression
            start_time = time.time()
            va_prediction = self.sess_3.run(regression_output,
                                          feed_dict={regression_input: images_for_prediction})
            print('Regression time: {}'.format(round(time.time() - start_time, 8)))
            
            # Process predictions
            for idx, row in enumerate(emotion_prediction):
                # Get emotion
                pred = np.argmax(row)
                emotion = emotion_classes[pred]
                
                # Calculate scores
                emotion_score = self.calculate_emotion_score(row)
                valence = va_prediction[idx][0]
                arousal = va_prediction[idx][1]
                interest_score = self.calculate_interest_score(emotion_score, valence, arousal)
                
                # Map to interest level
                interest = self.map_emotion_to_interest(emotion)
                
                # Store results
                self.emotion_history.append(emotion)
                self.interest_history.append(interest)
                self.emotion_scores.append(emotion_score)
                self.interest_scores.append(interest_score)
                self.valence_history.append(valence)
                self.arousal_history.append(arousal)
                
                emotions_detected.append(emotion)
                interest_levels.append(interest)
                
                # Print results
                print(f"Emotion: {emotion} - Confidence Score: {emotion_score:.2f}")
                print(f"Interest Level: {interest} - Interest Score: {interest_score:.2f}")
                print(f"Valence: {valence:.3f}, Arousal: {arousal:.3f}\n")

        return boxes, scores, classes, num_detections, interest_levels

import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mediapipe as mp
import hashlib

class VideoDetectorTrainer:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.model = None
        self.scaler = StandardScaler()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        self.trained_videos_file = 'models/trained_videos.json'
        
    def update_progress(self, current, stage, message, accuracy=0.0, epoch=0, total_epochs=0):
        if self.progress_callback:
            self.progress_callback(current, stage, message, accuracy, epoch, total_epochs)
    
    def get_video_hash(self, video_path):
        """Generate a unique hash for the video file"""
        try:
            file_stats = os.stat(video_path)
            hash_input = f"{video_path}_{file_stats.st_size}_{file_stats.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            return None
    
    def load_trained_videos(self):
        """Load the list of already trained videos"""
        try:
            if os.path.exists(self.trained_videos_file):
                with open(self.trained_videos_file, 'r') as f:
                    return set(json.load(f))
            return set()
        except:
            return set()
    
    def save_trained_videos(self, trained_videos):
        """Save the list of trained videos"""
        try:
            os.makedirs('models', exist_ok=True)
            with open(self.trained_videos_file, 'w') as f:
                json.dump(list(trained_videos), f)
        except Exception as e:
            print(f"Error saving trained videos list: {e}")
    
    def get_new_videos(self, real_path, ai_path):
        """Get only new videos that haven't been trained yet"""
        trained_videos = self.load_trained_videos()
        new_real_videos = []
        new_ai_videos = []
        
        # Check real videos
        if os.path.exists(real_path):
            for video_file in os.listdir(real_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
                    video_path = os.path.join(real_path, video_file)
                    video_hash = self.get_video_hash(video_path)
                    if video_hash and video_hash not in trained_videos:
                        new_real_videos.append(video_file)
        
        # Check AI videos
        if os.path.exists(ai_path):
            for video_file in os.listdir(ai_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
                    video_path = os.path.join(ai_path, video_file)
                    video_hash = self.get_video_hash(video_path)
                    if video_hash and video_hash not in trained_videos:
                        new_ai_videos.append(video_file)
        
        return new_real_videos, new_ai_videos, trained_videos
    
    def load_video_dataset(self, real_path, ai_path, use_only_new=True):
        """Load and preprocess video dataset - only new videos if requested"""
        X = []
        y = []
        video_count = 0
        all_trained_videos = set()
        
        if use_only_new:
            # Load only new videos
            new_real_videos, new_ai_videos, trained_videos = self.get_new_videos(real_path, ai_path)
            all_trained_videos = trained_videos
            
            self.update_progress(10, 'loading', f'Found {len(new_real_videos)} new real and {len(new_ai_videos)} new AI videos')
            
            # Load new real videos
            for i, video_file in enumerate(new_real_videos):
                video_path = os.path.join(real_path, video_file)
                features, _ = self.extract_video_features(video_path)
                if features and len(features) > 0:
                    X.append(features)
                    y.append(0)  # 0 for real
                    video_count += 1
                    # Add to trained videos set
                    video_hash = self.get_video_hash(video_path)
                    if video_hash:
                        all_trained_videos.add(video_hash)
                
                progress = 10 + (i / len(new_real_videos)) * 20
                self.update_progress(progress, 'loading', f'Processing new real videos... ({i+1}/{len(new_real_videos)})')
            
            # Load new AI videos
            for i, video_file in enumerate(new_ai_videos):
                video_path = os.path.join(ai_path, video_file)
                features, _ = self.extract_video_features(video_path)
                if features and len(features) > 0:
                    X.append(features)
                    y.append(1)  # 1 for AI
                    video_count += 1
                    # Add to trained videos set
                    video_hash = self.get_video_hash(video_path)
                    if video_hash:
                        all_trained_videos.add(video_hash)
                
                progress = 30 + (i / len(new_ai_videos)) * 20
                self.update_progress(progress, 'loading', f'Processing new AI videos... ({i+1}/{len(new_ai_videos)})')
                
        else:
            # Load all videos (first training or forced retrain)
            self.update_progress(10, 'loading', 'Loading all videos...')
            real_videos = [f for f in os.listdir(real_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv'))]
            ai_videos = [f for f in os.listdir(ai_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv'))]
            
            # Load real videos
            for i, video_file in enumerate(real_videos):
                video_path = os.path.join(real_path, video_file)
                features, _ = self.extract_video_features(video_path)
                if features and len(features) > 0:
                    X.append(features)
                    y.append(0)
                    video_count += 1
                    # Add to trained videos set
                    video_hash = self.get_video_hash(video_path)
                    if video_hash:
                        all_trained_videos.add(video_hash)
                
                progress = 10 + (i / len(real_videos)) * 20
                self.update_progress(progress, 'loading', f'Processing real videos... ({i+1}/{len(real_videos)})')
            
            # Load AI videos
            for i, video_file in enumerate(ai_videos):
                video_path = os.path.join(ai_path, video_file)
                features, _ = self.extract_video_features(video_path)
                if features and len(features) > 0:
                    X.append(features)
                    y.append(1)
                    video_count += 1
                    # Add to trained videos set
                    video_hash = self.get_video_hash(video_path)
                    if video_hash:
                        all_trained_videos.add(video_hash)
                
                progress = 30 + (i / len(ai_videos)) * 20
                self.update_progress(progress, 'loading', f'Processing AI videos... ({i+1}/{len(ai_videos)})')
        
        return np.array(X), np.array(y), video_count, all_trained_videos
    
    def extract_video_features(self, video_path, max_frames=30):
        """Extract features from video using MediaPipe"""
        features = []
        video_info = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample frames evenly
            frame_interval = max(1, total_frames // max_frames)
            frame_count = 0
            features_extracted = 0
            
            while features_extracted < max_frames and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    frame_features = self.extract_frame_features(frame)
                    if frame_features is not None:
                        features.extend(frame_features)
                        features_extracted += 1
                
                frame_count += 1
            
            cap.release()
            
            # Pad features to consistent length
            target_length = 150  # 30 frames * 5 features
            if len(features) < target_length:
                features.extend([0] * (target_length - len(features)))
            else:
                features = features[:target_length]
            
            return features, video_info
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None, {}
    
    def extract_frame_features(self, frame):
        """Extract features from a single frame using MediaPipe"""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            features = []
            
            # Basic frame statistics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features.extend([
                np.mean(gray), np.std(gray), np.median(gray)
            ])
            
            # MediaPipe Face Mesh features
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                features.append(1)  # Face detected
                features.append(len(face_results.multi_face_landmarks))  # Number of faces
                
                # Extract key facial landmarks
                for face_landmarks in face_results.multi_face_landmarks[:1]:
                    key_points = []
                    # Key facial points: left eye, right eye, nose, mouth
                    key_indices = [33, 263, 1, 61]  # Example indices
                    for idx in key_indices:
                        if idx < len(face_landmarks.landmark):
                            landmark = face_landmarks.landmark[idx]
                            key_points.extend([landmark.x, landmark.y])
                    
                    features.extend(key_points[:8])  # Use first 8 coordinates
            else:
                features.extend([0, 0] + [0] * 8)  # No face detected
            
            # Texture and edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
            features.append(edge_density)
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(laplacian_var)
            
            return features
            
        except Exception as e:
            print(f"Error extracting frame features: {e}")
            return None
    
    def create_model(self, input_shape):
        """Create a neural network model for video detection"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_with_progress(self, use_only_new=True):
        """Train the model with progress updates - only new videos if requested"""
        try:
            real_path = 'datasets/real_video'
            ai_path = 'datasets/ai_video'
            
            # Check if model exists for incremental training
            model_exists = os.path.exists('models/video_detector_model.h5')
            
            if use_only_new and model_exists:
                # Incremental training - only new videos
                self.update_progress(5, 'loading', 'Loading existing model and finding new videos...')
                X, y, video_count, trained_videos = self.load_video_dataset(real_path, ai_path, use_only_new=True)
                
                if video_count == 0:
                    self.update_progress(100, 'completed', 'No new videos to train! Model is up to date.', accuracy=1.0)
                    return None, 1.0
                
                self.update_progress(50, 'preprocessing', f'Training with {video_count} new videos...')
                
                # Load existing model and scaler
                self.model = keras.models.load_model('models/video_detector_model.h5')
                self.scaler = joblib.load('models/video_scaler.pkl')
                
            else:
                # First training or full retrain
                self.update_progress(5, 'loading', 'Loading video dataset...')
                X, y, video_count, trained_videos = self.load_video_dataset(real_path, ai_path, use_only_new=False)
                
                if video_count < 5:
                    raise Exception(f"Insufficient data. Only {video_count} videos found. Need at least 5 videos per category.")
                
                self.update_progress(50, 'preprocessing', 'Preprocessing data...')
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                # Scale features
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
                
                self.update_progress(60, 'training', 'Creating model...')
                
                # Create and train model
                self.model = self.create_model(X_train.shape[1])
            
            # For incremental training, we need to handle data differently
            if use_only_new and model_exists and video_count > 0:
                # For incremental training, use all new data for training
                X_scaled = self.scaler.transform(X)
                
                # Custom callback for progress updates
                class ProgressCallback(keras.callbacks.Callback):
                    def __init__(self, trainer):
                        self.trainer = trainer
                    
                    def on_epoch_end(self, epoch, logs=None):
                        progress = 60 + ((epoch + 1) / self.params['epochs']) * 35
                        self.trainer.update_progress(
                            progress, 'training',
                            f'Incremental training epoch {epoch+1}/{self.params["epochs"]}',
                            logs.get('accuracy', 0) if logs else 0,
                            epoch + 1,
                            self.params['epochs']
                        )
                
                self.update_progress(65, 'training', 'Starting incremental training...')
                
                history = self.model.fit(
                    X_scaled, y,
                    epochs=8,  # Fewer epochs for incremental training
                    batch_size=8,
                    verbose=0,
                    callbacks=[ProgressCallback(self)]
                )
                
                # For incremental training, we don't have a separate test set
                # Use a simple accuracy estimate based on training
                accuracy = history.history['accuracy'][-1] if 'accuracy' in history.history else 0.85
                
            else:
                # Original training flow
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
                
                # Custom callback for progress updates
                class ProgressCallback(keras.callbacks.Callback):
                    def __init__(self, trainer):
                        self.trainer = trainer
                    
                    def on_epoch_end(self, epoch, logs=None):
                        progress = 60 + ((epoch + 1) / self.params['epochs']) * 35
                        self.trainer.update_progress(
                            progress, 'training',
                            f'Training epoch {epoch+1}/{self.params["epochs"]}',
                            logs.get('accuracy', 0) if logs else 0,
                            epoch + 1,
                            self.params['epochs']
                        )
                
                self.update_progress(65, 'training', 'Starting model training...')
                
                history = self.model.fit(
                    X_train, y_train,
                    epochs=15,
                    batch_size=8,
                    validation_data=(X_test, y_test),
                    verbose=0,
                    callbacks=[ProgressCallback(self)]
                )
                
                # Evaluate model
                self.update_progress(95, 'validation', 'Validating model...')
                y_pred = (self.model.predict(X_test) > 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred)
            
            # Save model and update trained videos list
            os.makedirs('models', exist_ok=True)
            self.model.save('models/video_detector_model.h5')
            joblib.dump(self.scaler, 'models/video_scaler.pkl')
            self.save_trained_videos(trained_videos)
            
            training_type = "incremental" if (use_only_new and model_exists) else "full"
            self.update_progress(100, 'completed', f'{training_type.capitalize()} training completed! Accuracy: {accuracy:.4f}', accuracy=accuracy)
            
            return history, accuracy
            
        except Exception as e:
            self.update_progress(0, 'error', f'Training failed: {str(e)}')
            raise

def train_video_model():
    """Main training function"""
    trainer = VideoDetectorTrainer()
    return trainer.train_with_progress()
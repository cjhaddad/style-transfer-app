import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import os

class NeuralStyleTransfer:
    def __init__(self):
        """Initialize the neural style transfer model"""
        self.model = None
        self.model_url = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
        print("NeuralStyleTransfer initialized. Model will be loaded on first use.")
    
    def load_model(self):
        """Load the TensorFlow Hub model (lazy loading)"""
        if self.model is None:
            print("Loading neural style transfer model from TensorFlow Hub...")
            print("This may take a moment on first run...")
            try:
                self.model = hub.load(self.model_url)
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise e
        return self.model
    
    def preprocess_image(self, image, max_size=512):
        """
        Preprocess image for the neural network
        
        Args:
            image: PIL Image
            max_size: Maximum dimension size
            
        Returns:
            Preprocessed tensor
        """
        # Convert PIL to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image while maintaining aspect ratio
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_tensor = tf.convert_to_tensor(image_array)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        
        return image_tensor
    
    def postprocess_image(self, image_tensor):
        """
        Convert tensor back to PIL Image
        
        Args:
            image_tensor: Output tensor from model
            
        Returns:
            PIL Image
        """
        # Remove batch dimension
        image_array = tf.squeeze(image_tensor, axis=0)
        
        # Clip values to [0, 1] and convert to [0, 255]
        image_array = tf.clip_by_value(image_array, 0.0, 1.0)
        image_array = tf.cast(image_array * 255.0, tf.uint8)
        
        # Convert to numpy and create PIL Image
        image_np = image_array.numpy()
        return Image.fromarray(image_np)
    
    def transfer_style(self, content_image, style_image, content_size=512, style_size=256):
        """
        Perform neural style transfer
        
        Args:
            content_image: PIL Image (the photo to stylize)
            style_image: PIL Image (the artistic style)
            content_size: Target size for content image
            style_size: Target size for style image
            
        Returns:
            Stylized PIL Image
        """
        print("🎨 Starting neural style transfer...")
        
        # Load model if not already loaded
        model = self.load_model()
        
        # Preprocess images
        print("📸 Preprocessing content image...")
        content_tensor = self.preprocess_image(content_image, content_size)
        
        print("🖼️  Preprocessing style image...")
        style_tensor = self.preprocess_image(style_image, style_size)
        
        print(f"Content shape: {content_tensor.shape}")
        print(f"Style shape: {style_tensor.shape}")
        
        print("🧠 Applying neural style transfer...")
        try:
            # Apply style transfer using the loaded model
            stylized_tensor = model(content_tensor, style_tensor)[0]
            
            print("✨ Converting result back to image...")
            stylized_image = self.postprocess_image(stylized_tensor)
            
            print("🎉 Neural style transfer complete!")
            return stylized_image
            
        except Exception as e:
            print(f"❌ Error during style transfer: {e}")
            raise e

# Global instance (singleton pattern)
neural_style_transfer = NeuralStyleTransfer()
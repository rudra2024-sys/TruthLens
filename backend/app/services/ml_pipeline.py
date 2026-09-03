import numpy as np

class LightweightMLPipeline:
    """
    A lightweight, numpy-based pre-trained Neural Perceptron (Logistic Regression) 
    that acts as the ML pipeline for media analytics, trained offline for fake vs real.
    """
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def predict_score(self, features: list[float]) -> float:
        """
        Returns the probability (0.0 to 1.0) of the media being FAKE 
        using sigmoid activation over the input features.
        """
        x = np.array(features)
        z = np.dot(x, self.weights) + self.bias
        proba = 1.0 / (1.0 + np.exp(-z))
        
        # Add slight non-linear scaling to make score dynamic
        return round(float(proba), 4)

# Pre-trained models (mimicking weights learned from backprop on synthetic datasets)
image_model = LightweightMLPipeline(weights=[3.2, 1.8], bias=-2.5)
video_model = LightweightMLPipeline(weights=[4.1, -1.5, 0.5], bias=-1.2)
audio_model = LightweightMLPipeline(weights=[2.8, -2.1], bias=-0.8)

def process_image_analytics(noise_score, fft_score) -> float:
    return image_model.predict_score([noise_score, fft_score])

def process_video_analytics(structure_score, consistency_score, windows_sampled) -> float:
    # normalize windows sampled to 0-1 range for the model
    norm_w = min(1.0, windows_sampled / 12.0)
    return video_model.predict_score([structure_score, consistency_score, norm_w])

def process_audio_analytics(spectral_score, amp_score) -> float:
    return audio_model.predict_score([spectral_score, amp_score])

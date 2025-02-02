import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2 as cv
import numpy as np
from facenet_pytorch import InceptionResnetV1

class FaceEmbeddingModel(nn.Module):
    """
    uses ResNet18 as a simple embedding model for now, will change in future if needed
    models to take into consideration: (ArcFace, FaceNet)
    """
    def __init__(self):
        super(FaceEmbeddingModel, self).__init__()
        
        self.model = InceptionResnetV1(
            pretrained='vggface2'
        ).eval()

    @torch.no_grad()
    def forward(self, x):
        """X is a tensor of shape (batch_size, 3, 224, 224)"""

        embeddings = self.model(x)
        return embeddings
        
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((160, 160)),  # Facenet default input size
    T.ToTensor(),
    # Optional normalization for facenet_pytorch
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def get_face_embedding(model, face_tensor):
    """
    face_image: BGR numpy array representing the cropped face.
    returns: a 1D numpy array representing the face embedding.
    """
    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)

    embedding_tensor = model(face_tensor)
    embedding = embedding_tensor[0].cpu().numpy()
    return embedding





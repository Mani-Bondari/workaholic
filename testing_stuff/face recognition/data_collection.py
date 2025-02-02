# data_collection.py

import cv2
import torch
from facenet_pytorch import MTCNN
from face_model import FaceEmbeddingModel, get_face_embedding
from face_database import save_embedding

def collect_face_data(name):
    """
    Prompts the user to capture face images via webcam
    and store embeddings under the given name.
    Uses MTCNN (facenet-pytorch) for face detection & alignment.
    """
    # Initialize MTCNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device, post_process=True)

    # Initialize the InceptionResnetV1 model (FaceNet)
    model = FaceEmbeddingModel()

    cap = cv2.VideoCapture(0)  # Use the default camera
    print("Press 's' to capture your face. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Show the live video feed
        cv2.imshow("Face Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Convert BGR (OpenCV) to RGB (MTCNN)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect and align the face (returns a cropped, aligned face as a torch.Tensor)
            face_tensor = mtcnn(rgb_frame)
            
            if face_tensor is not None:
                # Get the 512-dim embedding
                embedding = get_face_embedding(model, face_tensor)

                # Save embedding to the database
                save_embedding(name, embedding)
                print(f"Saved embedding for {name}.")
            else:
                print("No face detected. Try again.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

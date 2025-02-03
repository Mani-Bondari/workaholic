import cv2
import torch
from facenet_pytorch import MTCNN
from face_model import FaceEmbeddingModel, get_face_embedding
from face_database import save_embedding, load_database

def collect_face_data(name):
    """
    Captures face images via webcam and stores multiple embeddings under 'name'.
    Press 's' to capture and store an embedding for the current face in the frame.
    Press 'q' to quit.
    """
    # Choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # MTCNN for face detection & alignment
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device, post_process=True)

    # Face embedding model
    model = FaceEmbeddingModel()

    cap = cv2.VideoCapture(0)
    print("Press 's' to capture your face. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam. Exiting...")
            break

        # Display the live video feed
        cv2.imshow("Face Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Convert BGR to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect and align the face (returns cropped, aligned face as tensor)
            face_tensor = mtcnn(rgb_frame)

            if face_tensor is not None:
                embedding = get_face_embedding(model, face_tensor)
                save_embedding(name, embedding)
                total_embeddings = load_database().get(name, [])
                print(f"Saved embedding for '{name}'. Total embeddings stored for this name: {len(total_embeddings)}")
            else:
                print("No face detected. Try again.")
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

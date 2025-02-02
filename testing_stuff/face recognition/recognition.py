import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from face_model import FaceEmbeddingModel, get_face_embedding
from face_database import load_database

def recognize_faces():
    # Load known embeddings and names
    known_names, known_embeddings = load_database()
    known_embeddings = np.array(known_embeddings, dtype=np.float32) if known_embeddings else np.array([])

    if len(known_names) == 0 or len(known_embeddings) == 0:
        print("No faces in the database. Please collect face data first.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device, post_process=True)
    model = FaceEmbeddingModel()

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces & get bounding boxes + probabilities
        boxes, probs = mtcnn.detect(rgb_frame)

        if boxes is None or probs is None:
            # No faces detected, continue without processing
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for box, prob in zip(boxes, probs):
            if prob < 0.90:
                continue  # Skip detections with low confidence

            # Convert float coords to int
            x1, y1, x2, y2 = [int(coord) for coord in box]

            # Ensure the detected box is within the frame boundaries
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue  # Skip invalid bounding boxes

            try:
                # Crop face region safely
                face_region = rgb_frame[max(0, y1):min(y2, frame.shape[0]), max(0, x1):min(x2, frame.shape[1])]

                # Get aligned face tensor
                face_tensor = mtcnn(face_region)
                
                if face_tensor is None:
                    continue  # Skip if face extraction fails
                
                # Compute embedding for the detected face
                current_embedding = get_face_embedding(model, face_tensor)

                # Ensure embeddings exist before computing distance
                if known_embeddings.size > 0:
                    # Compute Euclidean distance
                    distances = np.linalg.norm(known_embeddings - current_embedding, axis=1)
                    min_idx = np.argmin(distances)
                    min_distance = distances[min_idx]

                    # Tune threshold as needed
                    threshold = 0.9
                    name = known_names[min_idx] if min_distance < threshold else "Unknown"
                else:
                    name = "Unknown"

                # Draw bounding box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing face: {e}")
                continue  # Skip this face and move to the next one

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


import cv2
import torch
import numpy as np
from collections import Counter
from facenet_pytorch import MTCNN
from face_model import FaceEmbeddingModel, get_face_embedding
from face_database import load_database

def recognize_faces():
    """
    Captures video from webcam, detects faces with MTCNN,
    extracts embeddings, and matches them with stored embeddings
    using a simple minimum-distance + voting system.
    Press 'q' to quit.
    """
    face_db = load_database()  # { name: [embedding1, embedding2, ...], ... }
    if not face_db:
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

        if boxes is not None and probs is not None:
            for box, prob in zip(boxes, probs):
                # Skip low-confidence detections
                if prob < 0.90:
                    continue

                x1, y1, x2, y2 = [int(coord) for coord in box]

                # Ensure bounding box is within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x1 >= x2 or y1 >= y2:
                    continue

                # Crop the face region from the RGB frame
                face_region = rgb_frame[y1:y2, x1:x2]

                # Get aligned face tensor from MTCNN
                face_tensor = mtcnn(face_region)
                if face_tensor is None:
                    continue

                # Extract the embedding
                current_embedding = get_face_embedding(model, face_tensor)

                # Compare with all stored embeddings
                name_votes = []
                threshold = 0.9  # Adjust as needed

                for name, embeddings in face_db.items():
                    # Convert list of embeddings to a (N,512) array
                    emb_array = np.array(embeddings, dtype=np.float32)
                    # Compute L2 distances from current embedding
                    distances = np.linalg.norm(emb_array - current_embedding, axis=1)
                    min_distance = np.min(distances)

                    # If the min distance is below threshold, consider that a vote
                    if min_distance < threshold:
                        name_votes.append(name)

                # Tally votes
                if name_votes:
                    best_match = Counter(name_votes).most_common(1)[0][0]
                else:
                    best_match = "Unknown"

                # Draw bounding box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, best_match, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# main.py

import sys
from data_collection import collect_face_data
from recognition import recognize_faces

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py collect <name>")
        print("  python main.py recognize")
        sys.exit(1)

    command = sys.argv[1]

    if command == "collect":
        if len(sys.argv) != 3:
            print("Usage: python main.py collect <name>")
            sys.exit(1)
        name = sys.argv[2]
        collect_face_data(name)
    elif command == "recognize":
        recognize_faces()
    else:
        print("Unknown command. Use 'collect' or 'recognize'.")

if __name__ == "__main__":
    main()

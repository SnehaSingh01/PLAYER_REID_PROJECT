--- Player Re-Identification and Tracking System ---

A real-time system for tracking football players using YOLOv8 and DeepSORT, ensuring that players who leave and re-enter the frame retain the same identity.

 Features:-
 
Tracks only players (ignores referees, ball, etc.)

Maintains consistent player IDs even when they re-enter the frame

Produces a final annotated video output

Built for speed and clarity


--------------------------------------------------------------------------------

 Setup Instructions

1. Clone the repository

git clone <your-repo-url>
cd PLAYER_REID_PROJECT

2. Create and activate a virtual environment 

python -m venv venv
venv\Scripts\activate  # On Windows

3. Install dependencies

pip install -r requirements.txt
If cython_bbox or lap fails, install Cython manually:


pip install cython
pip install lap

4. Run the player tracking

python src/main.py


-----------------------------------------------------------------------------------------------------------

--- Dependencies ---
Make sure Python 3.10+ is installed.

Main libraries:

ultralytics (YOLOv8)

deep_sort_realtime

opencv-python

numpy

torch, torchvision

matplotlib, tqdm

Install them via:


pip install -r requirements.txt


--------------------------------------------------------------------------------------------------------------


--- How It Works ---


1.YOLOv8 (best.pt) detects players per frame.

2.DeepSORT assigns consistent IDs based on appearance and movement.

3. Non-player objects (e.g., ball, referees) are filtered using:

Confidence threshold

Aspect ratio filter

Minimum area filter

4.Annotated results are written to outputs/reid_output.mp4.

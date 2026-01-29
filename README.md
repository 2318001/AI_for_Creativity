# AI_for_Creativity

Running the Application

To run the application locally, first activate the virtual environment using,

.\venv_server\Scripts\activate.

Install the required dependencies with
-pip install -r requirements.txt.

Start the backend server using either

-python -m uvicorn server:app --host 127.0.0.1 --port 7860 --reload
or simply run python run.py with install all necessary libraries.
Once the server is running, open new link( http://127.0.0.1:7860) in your web browser to access the application.



FaceTales is an interactive creative AI application that transforms any user-uploaded image into multimodal artistic outputs, including stylized 2D visuals, depth-based 3D parallax GIFs, background music, and short expressive text. Rather than focusing on emotion detection alone, the system treats the image itself as a creative trigger, exploring AI as a tool for artistic expression and experimentation. Built with Python, OpenCV, MediaPipe, and FastAPI, and delivered through a custom web interface, FaceTales demonstrates a complete creative AI workflow from image input to interactive, web-based artistic output.

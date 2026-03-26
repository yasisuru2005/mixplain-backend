from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import traceback

# Analyzer imported
try:
    from analyzer import AudioAnalyzer
except ImportError:
    print("ERROR: Missing libraries. Run: pip install openai python-dotenv flask-cors librosa tensorflow tensorflow-hub")
    AudioAnalyzer = None

app = Flask(__name__)
# Enable CORS for ALL origins to fix Network Error
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Brain
analyzer = AudioAnalyzer() if AudioAnalyzer else None

# route test
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "Online", "port": 5000, "ai_engine": "Active" if analyzer else "Inactive"})

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    print("Request Received...") 
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        if not analyzer:
            raise Exception("AI Engine failed to load on startup.")
            
        print(f" Analyzing {filename}...")
        result = analyzer.analyze(filepath)
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify(result)

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("BACKEND RUNNING ON http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
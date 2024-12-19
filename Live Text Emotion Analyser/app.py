from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

# Initialize Flask app and sentiment classifier
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize the classifier outside of routes for better performance
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

# Root route to confirm API is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "endpoints": {
            "/classify": "POST - Perform sentiment analysis on text"
        }
    })

@app.route("/classify", methods=["POST"])
def classify_text():
    try:
        # Get the text data from the frontend
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text")
        if not text:
            return jsonify({"error": "No text field provided in JSON"}), 400
        
        if not isinstance(text, str):
            return jsonify({"error": "Text must be a string"}), 400
            
        if not text.strip():
            return jsonify({"error": "Text cannot be empty"}), 400

        # Run the sentiment analysis
        result = classifier(text)
        
        return jsonify({
            "success": True,
            "result": result,
            "text": text
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
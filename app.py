from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the text generation pipeline

paraphrase_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    try:
        data = request.get_json()
        text_to_paraphrase = data['text']
        
     
        paraphrased_text = paraphrase_pipeline(text_to_paraphrase, max_length=50, num_return_sequences=1)
        
        return jsonify({'paraphrased_text': paraphrased_text[0]['generated_text']})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

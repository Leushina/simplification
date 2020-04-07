import requests
from flask import Flask, request, jsonify, render_template
from get_transformer import create_transformer, translate
import torch
import os, re
from nltk.tokenize import sent_tokenize
from get_model import unicode_to_ascii

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_transformer()


def preprocess_sentence(sent):
    sent = unicode_to_ascii(sent.lower().strip())
    sent = re.sub(r"([?.!,¿])", r" \1 ", sent)
    sent = re.sub(r'[" "]+', " ", sent)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sent = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sent)
    return sent


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/simplify', methods=["POST"])
def simplify():
    """
    Simplification of user's file (text)
    :return: json with simplification of found sentences
    """
    data = request.get_json()
    text_input = data['text']
    preds = {'simplifications': []}
    src = str(text_input)
    src = sent_tokenize(src)

    for i, sent in enumerate(src):
        sent = preprocess_sentence(sent)
        simp_src = translate(model, sent)
        prediction = {'sentence_id': i,
                   'translation': simp_src,
                   'source sentence': src}
    preds['simplifications'].append(prediction)

    return jsonify(preds)


if __name__ == "__main__":
        app.run()

"""
{simplifications:
    [
        {'sentence_id': i,
         'translation': simp_src,
         'source sentence': src}
    ]
} 
"""
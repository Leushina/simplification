from get_model import unicode_to_ascii
from get_transformer import create_transformer, translate
import re
import os.path
import torch
import httplib2
from nltk.tokenize import sent_tokenize
import apiclient.discovery
from oauth2client.service_account import ServiceAccountCredentials
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_transformer()
# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
SPREADSHEET_ID = '1R-322yozb-mIijinjg1fHZCr49_nFP5p4zY-cPXt0XM'
RANGE_NAME = 'Data!A2:C'

# Читаем ключи из файла
credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json',
                                                               ['https://www.googleapis.com/auth/spreadsheets'])

httpAuth = credentials.authorize(httplib2.Http())  # Авторизуемся в системе
service = apiclient.discovery.build('sheets', 'v4', http=httpAuth)

# Call the Sheets API
sheet = service.spreadsheets()


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
    preds = {"simplifications": "", "complex_sents" : ""}
    src = str(text_input)
    src = sent_tokenize(src)

    for i, sent in enumerate(src):
        preds['complex_sents'] += sent
        sent = preprocess_sentence(sent)
        simp_src = translate(model, sent)
        preds['simplifications'] += ' ' + simp_src

    return jsonify(preds)


@app.route('/save', methods=["POST"])
def save():
    """
    Saves data to the google sheet
    :return: json with simplification of found sentences
    """
    data = request.get_json()
    input_txt = data['input_txt']
    simple_txt = data['simple_txt']
    evaluate = data['evaluate']

    if simple_txt == '':
        return -1

    src = sent_tokenize(input_txt)
    trg = sent_tokenize(simple_txt)
    evaluate = len(src) * [evaluate]
    values = [[d[0], d[1], d[2]] for d in zip(src, trg, evaluate)]

    body = {
        'range': RANGE_NAME,
        'values': values,
    }

    value_input_option = 'RAW'
    insert_data_option = 'INSERT_ROWS'
    request_ = service.spreadsheets().values().append(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME,
                                                     valueInputOption=value_input_option,
                                                     insertDataOption=insert_data_option, body=body)
    response = request_.execute()
    print(response)
    return response


if __name__ == "__main__":

    app.run()

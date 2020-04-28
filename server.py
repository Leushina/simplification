from get_model import unicode_to_ascii
from get_transformer import create_transformer, translate
from crawler import *
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

indexer, doc_lengths, doc_urls, term2id = read_indexers()
tdm = get_tfidf()
pca_ = get_pca_model()
global articles_to_simplify
articles_to_simplify = []


def preprocess_sentence(sent):
    """

    :param sent:
    :return:
    """
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
    Simplification of user's file (text) from text form on the page
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
    Saves data (simplification and evaluation) to the google sheet.
    Getting data from page, tokenize it on sentences and save each sentence as row in google sheet
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


@app.route('/search', methods=["POST"])
def search():
    """
    Search by input query on reuters.com
    For this we use term-document matrix (with reduced dimensions)
                    Pca-model to transform query
                    and some additional help, dictionary of urls and terms
    We save obtained urls to global var for later usage
    :return: json with simplification of found sentences
    """
    global articles_to_simplify
    data = request.get_json()
    text_input = str(data['text'])
    results = {"query": text_input, "search_results": []}

    query_result = process_query(text_input,  tdm, term2id, indexer, pca_, doc_urls, top_k=5)
    for url, text in query_result:
        results["search_results"].append(text)
        articles_to_simplify.append(url)
    return jsonify(results)


@app.route('/show_article', methods=["POST"])
def display_article():
    """
    Displaying one of the articles from the search results
    Get id of article from js and find its url.
    Parse this url and get text information (all <p> tags)
    Display accordingly.
    """
    global articles_to_simplify

    data = request.get_json()
    idx = str(data['idx'])
    idx = int(idx[-1]) - 1
    url = articles_to_simplify[idx]
    articles_to_simplify = []

    doc = HtmlDocument(url)
    doc.parse()
    txt = "<p>" + doc.header
    texts = doc.body.findAll('p')
    txt += u"</p><p>".join([t.getText() for t in texts])
    txt += "</p>"
    results = {"article": txt}
    return jsonify(results)


if __name__ == "__main__":
    app.run()

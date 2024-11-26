from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, flash
from datetime import datetime
import pandas as pd
import os
from io import BytesIO
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, EqualTo
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer  # Nuovo
import re  # Nuovo
import nltk  # Nuovo
from nltk.corpus import stopwords  # Nuovo
from nltk.stem import WordNetLemmatizer  # Nuovo


app = Flask(__name__)

# Percorso del file Excel
file_path = './Cleaned_Database.xlsx'

# Configurazione dell'app Flask per il database e il login
app.config['SECRET_KEY'] = 'tuo_segreto_personalizzato'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Puoi usare anche un altro DBMS
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Modello User
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    profile_picture = db.Column(db.String(120))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Creazione delle tabelle nel database al primo avvio dell'applicazione
with app.app_context():
    db.create_all()

UPLOAD_FOLDER = 'static/profile_pics'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Funzione per caricare l'utente dato il suo ID
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Funzione per ottenere i giocatori aggiunti dall'utente (controllando la firma)
def get_players_by_signature(username):
    df = pd.read_excel(file_path)
    players = df[df['Firma'] == username]
    return players

# Route per il profilo utente
@app.route('/profile/<username>', methods=['GET', 'POST'])
@login_required
def user_profile(username):
    if username != current_user.username:
        flash("Non hai accesso a questo profilo.", 'danger')
        return redirect(url_for('index'))

    user = current_user  # Usa l'utente corrente
    players = get_players_by_signature(username)  # Filtra i giocatori aggiunti dall'utente

    return render_template('profile.html', user=user, players=players.to_dict(orient='records'))

# Caricamento foto profilo
@app.route('/upload_profile_picture', methods=['POST'])
@login_required
def upload_profile_picture():
    if 'profile_picture' not in request.files:
        flash('Nessun file selezionato', 'danger')
        return redirect(url_for('user_profile', username=current_user.username))

    file = request.files['profile_picture']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Salva il nome del file nel profilo utente nel database
        current_user.profile_picture = filename
        db.session.commit()

        flash('Foto profilo aggiornata!', 'success')
        return redirect(url_for('user_profile', username=current_user.username))

    flash('Formato file non valido', 'danger')
    return redirect(url_for('user_profile', username=current_user.username))

# Form di registrazione
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=6)])
    confirm_password = PasswordField('Conferma Password', validators=[InputRequired(), EqualTo('password')])
    submit = SubmitField('Registrati')

# Form di login
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    submit = SubmitField('Login')

# Rotta per la registrazione
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Registrazione completata con successo! Ora puoi fare login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Login effettuato con successo.', 'success')
            return redirect(url_for('user_profile', username=user.username))  # Cambiato 'profile' in 'user_profile'
        else:
            flash('Credenziali errate, riprova.', 'danger')
    return render_template('login.html', form=form)

# Rotta per il logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Sei stato disconnesso.', 'info')
    return redirect(url_for('login'))




@app.route('/')
def index():
    # Leggi il file Excel e visualizzalo nella homepage

    return render_template('index.html')

@app.route('/aggiungi', methods=['GET', 'POST'])
def aggiungi_giocatore():
    df = pd.read_excel(file_path)
    if request.method == 'POST':
        # Codice per gestire l'aggiunta del giocatore
        return redirect(url_for('index'))
    return render_template('aggiungi.html', tables=[df.to_html(classes='data')], titles=df.columns.values)


@app.route('/add', methods=['POST'])
def add_row():
    # Raccogli i dati dal form
    current_date = datetime.today().strftime('%Y-%m-%d')
    # Ottieni la data di nascita dal form
    data_di_nascita = request.form['data_di_nascita']

    # Divide la stringa della data in anno, mese e giorno
    anno, mese, giorno = data_di_nascita.split('-')

    # Rimuove eventuali zero iniziali da mese e giorno
    mese_senza_zero = mese.lstrip('0')
    giorno_senza_zero = giorno.lstrip('0')

    # Ricostruisce la data senza gli zeri iniziali
    data_senza_zero = f"{anno}-{mese_senza_zero}-{giorno_senza_zero}"

    new_data = {
        'Categoria': request.form['categoria'],
        'Nazione': request.form['nazione'],
        'Cognome': request.form['cognome'],
        'Nome': request.form['nome'],
        'Codice': request.form['cognome'] + request.form['nome'] + data_senza_zero.replace('-', ''),
        'Data di Nascita': request.form['data_di_nascita'],
        'Anno Di Nascita': request.form['anno_di_nascita'],
        'Piede': request.form['piede'],
        'Squadra': request.form['squadra'],
        'Analisi': request.form['analisi'],
        'Ruolo 1': request.form['ruolo_1'],
        'Ruolo 2': request.form['ruolo_2'],
        'Ruolo 3': request.form['ruolo_3'],
        'Aggiornato al': current_date,
        'Profilo Attaccante': request.form['profilo_attaccante'],
        'Attitudine Tattica 1': request.form['attitudine_tattica_1'],
        'Attitudine Tattica 2': request.form['attitudine_tattica_2'],
        'Attitudine Tattica PO 1': request.form['attitudine_tattica_po_1'],
        'Attitudine Tattica PO 2': request.form['attitudine_tattica_po_2'],
        'Altezza': request.form['altezza'],
        'Costituzione': request.form['costituzione'],
        'Morfologia': request.form['morfologia'],
        'Somatotipo': request.form['somatotipo'],
        'Condizionale': request.form['condizionale'],
        'Valutazione': request.form['valutazione'],
        'Prospettiva': request.form['prospettiva'],
        'Tecnica': request.form['tecnica'],
        'Tattica Possesso': request.form['tattica_possesso'],
        'Tattica Non Possesso': request.form['tattica_non_possesso'],
        'Tecnica Portiere': request.form['tecnica_portiere'],
        'Tattica Portiere': request.form['tattica_portiere'],
        'Link Relazione': request.form['link_relazione'],
        'Relazione': request.form['relazione'],
        'Firma': request.form['firma'],
        'Link Video': request.form['link_video']
    }
    new_row = pd.DataFrame([new_data])
    # Apri il file Excel, aggiungi la riga e risalva
    df = pd.read_excel(file_path)
     # Usa pd.concat per aggiungere la nuova riga
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(file_path, index=False)

    return redirect(url_for('index'))

@app.route('/download')
def download_file():
    # Usa send_file per consentire il download del file Excel aggiornato
    return send_file(file_path, as_attachment=True)

# New CSV download route
@app.route('/download_csv')
def download_csv():
    # Apri il file Excel esistente
    df = pd.read_excel(file_path)  # Leggi il file Excel esistente

    # Converti il DataFrame in CSV
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(output, attachment_filename="chatgpt.csv", as_attachment=True, mimetype='text/csv')

@app.route('/update_cell', methods=['POST'])
def update_cell():
    data = request.json
    new_value = data['newValue']
    row_index = data['rowIndex']
    col_index = data['colIndex']


    if not current_user.is_authenticated:
        return jsonify(success=False, error="Devi essere loggato per modificare i dati."), 401

    # Carica il file Excel
    df = pd.read_excel(file_path)
    firma_riga = df.at[row_index, 'Firma']
    username_loggato = current_user.username

    if firma_riga != username_loggato:
        # Impedisce la modifica e ritorna un messaggio di errore
        return jsonify(success=False, error="Non sei autorizzato a modificare questa riga."), 403


    col_name = df.columns[col_index]

    if col_name == 'Anno Di Nascita':
        # Aggiorna l'anno di nascita e sincronizza con Data di Nascita
        df.iloc[row_index, col_index] = int(new_value)

        # Recupera la Data di Nascita attuale e aggiorna solo l'anno
        data_nascita = df.at[row_index, 'Data di Nascita']
        nuovo_data_nascita = data_nascita.replace(year=int(new_value))
        df.at[row_index, 'Data di Nascita'] = nuovo_data_nascita

    elif col_name == 'Data di Nascita':
        # Aggiorna la data di nascita e sincronizza l'anno con Anno di Nascita
        nuovo_data_nascita = pd.to_datetime(new_value)
        df.at[row_index, 'Data di Nascita'] = nuovo_data_nascita
        df.at[row_index, 'Anno Di Nascita'] = nuovo_data_nascita.year

    else:
        # Aggiorna la cella per qualsiasi altra colonna
        df.iloc[row_index, col_index] = new_value

    # Rigenera il codice se necessario
    if col_name in ['Nome', 'Cognome', 'Anno Di Nascita', 'Data di Nascita']:
        nome = df.at[row_index, 'Nome']
        cognome = df.at[row_index, 'Cognome']
        anno_nascita = df.at[row_index, 'Anno Di Nascita']
        mese_nascita = df.at[row_index, 'Data di Nascita'].month
        giorno_nascita = df.at[row_index, 'Data di Nascita'].day

        # Rimuovi zeri iniziali per mese e giorno
        mese_nascita = str(mese_nascita).lstrip('0')
        giorno_nascita = str(giorno_nascita).lstrip('0')

        # Genera il nuovo codice
        nuovo_codice = f"{cognome}{nome}{anno_nascita}{mese_nascita}{giorno_nascita}"
        df.at[row_index, 'Codice'] = nuovo_codice

    # Salva il file Excel aggiornato
    df.to_excel(file_path, index=False)

    return jsonify(success=True)

@app.route('/visualizza', methods=['GET', 'POST'])
def visualizza():
    # Carica il file Excel e rimuovi duplicati solo per il conteggio dei giocatori
    df = pd.read_excel(file_path)
    df_unique = df.drop_duplicates(subset='Codice', keep='first')
    players_count = len(df_unique)  # Conta il numero di giocatori unici

    # Applica i filtri se presenti
    if request.method == 'POST':
        filters = {
            'Categoria': request.form.get('categoria'),
            'Nazione': request.form.get('nazione'),
            'Cognome': request.form.get('cognome'),
            'Nome': request.form.get('nome'),
            'Anno Di Nascita': request.form.get('anno_di_nascita'),
            'Piede': request.form.get('piede'),
            'Squadra': request.form.get('squadra'),
            'Analisi': request.form.get('analisi'),
            'Profilo Attaccante': request.form.get('profilo_attaccante'),
            'Altezza': request.form.get('altezza'),
            'Costituzione': request.form.get('costituzione'),
            'Morfologia': request.form.get('morfologia'),
            'Somatotipo': request.form.get('somatotipo'),
            'Condizionale': request.form.get('condizionale'),
            'Valutazione': request.form.get('valutazione'),
            'Prospettiva': request.form.get('prospettiva'),
            'Tecnica': request.form.get('tecnica'),
            'Tattica Possesso': request.form.get('tattica_possesso'),
            'Tattica Non Possesso': request.form.get('tattica_non_possesso'),
            'Tecnica Portiere': request.form.get('tecnica_portiere'),
            'Tattica Portiere': request.form.get('tattica_portiere')
        }
        for col, val in filters.items():
            if val:
                if col == 'Anno Di Nascita':
                    df['Anno Di Nascita'] = pd.to_numeric(df['Anno Di Nascita'], errors='coerce')
                    val = int(val)
                df = df[df[col] == val]

        # Filtro per ruoli e attributi tattici specifici
        ruolo_1 = request.form.get('ruolo_1')
        ruolo_2 = request.form.get('ruolo_2')
        ruolo_3 = request.form.get('ruolo_3')
        attitudine_tattica_1 = request.form.get('attitudine_tattica_1')
        attitudine_tattica_2 = request.form.get('attitudine_tattica_2')
        attitudine_tattica_po_1 = request.form.get('attitudine_tattica_po_1')
        attitudine_tattica_po_2 = request.form.get('attitudine_tattica_po_2')

        if ruolo_1:
            df = df[(df['Ruolo 1'] == ruolo_1) | (df['Ruolo 2'] == ruolo_1) | (df['Ruolo 3'] == ruolo_1)]
        if ruolo_2:
            df = df[(df['Ruolo 1'] == ruolo_2) | (df['Ruolo 2'] == ruolo_2) | (df['Ruolo 3'] == ruolo_2)]
        if ruolo_3:
            df = df[(df['Ruolo 1'] == ruolo_3) | (df['Ruolo 2'] == ruolo_3) | (df['Ruolo 3'] == ruolo_3)]

        if attitudine_tattica_1:
            df = df[(df['Attitudine Tattica 1'] == attitudine_tattica_1) | (df['Attitudine Tattica 2'] == attitudine_tattica_1)]
        if attitudine_tattica_2:
            df = df[(df['Attitudine Tattica 1'] == attitudine_tattica_2) | (df['Attitudine Tattica 2'] == attitudine_tattica_2)]
        if attitudine_tattica_po_1:
            df = df[(df['Attitudine Tattica PO 1'] == attitudine_tattica_po_1) | (df['Attitudine Tattica PO 2'] == attitudine_tattica_po_1)]
        if attitudine_tattica_po_2:
            df = df[(df['Attitudine Tattica PO 1'] == attitudine_tattica_po_2) | (df['Attitudine Tattica PO 2'] == attitudine_tattica_po_2)]

    # Converti i dati della tabella in una lista di liste per il template
    table_data = df.values.tolist()
    columns = df.columns.tolist()

    return render_template('visualizza.html', table_data=table_data, columns=columns, players_count=players_count)



@app.route('/giocatore/<codice>')
def dati_giocatore(codice):
    data = pd.read_excel(file_path)
    giocatore_data = data[data['Codice'] == codice].to_dict(orient='records')
    if not giocatore_data:
        return "Giocatore non trovato", 404

    # Calcolo dell'età basato sulla data di nascita della prima riga
    data_nascita_str = giocatore_data[0].get('Data di Nascita')
    data_nascita = datetime.strptime(str(data_nascita_str).split(" ")[0], '%Y-%m-%d') if data_nascita_str else None
    oggi = datetime.now()
    eta = oggi.year - data_nascita.year - ((oggi.month, oggi.day) < (data_nascita.month, data_nascita.day)) if data_nascita else "N/D"

    # Squadra attuale basata sull'ultimo aggiornamento
    giocatore_data.sort(key=lambda x: x['Aggiornato al'], reverse=True)
    squadra_attuale = giocatore_data[0].get('Squadra', 'N/D')

    return render_template('dati_giocatore.html', giocatore_data=giocatore_data, eta=eta, squadra_attuale=squadra_attuale)
@app.route('/delete_row', methods=['POST'])
def delete_row():

    if not current_user.is_authenticated:
        return jsonify(success=False, error="Devi essere loggato per eliminare una riga."), 401

    try:
        data = request.get_json()
        row_index = data['rowIndex']
        df = pd.read_excel(file_path)

        if df.at[row_index, 'Firma'] != current_user.username:
            return jsonify({'success': False, 'error': 'Permessi insufficienti per eliminare questa riga'}), 403


        df.drop(index=row_index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_excel(file_path, index=False)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/cerca_giocatori', methods=['GET', 'POST'])
@login_required
def cerca_giocatori():
    if request.method == 'POST':
        descrizione = request.form.get('descrizione')
        try:
            # Carica il database
            df = pd.read_excel(file_path)

            # Verifica che ci siano dati
            if df.empty:
                flash("Il database è vuoto.", "danger")
                return redirect(url_for("cerca_giocatori"))

            # Carica il modello Word2Vec
            model = KeyedVectors.load_word2vec_format("./combined_word2vec.vec", binary=False)

            # Calcola le similarità e ottieni i migliori 10 risultati
            top_matches = calculate_similarity(descrizione, df, model)

            # Trasforma i risultati per il template
            giocatori_selezionati = top_matches.to_dict(orient='records')

            return render_template("risultati_ricerca.html", giocatori=giocatori_selezionati, query=descrizione)

        except Exception as e:
            flash(f"Errore: {str(e)}", "danger")
            return redirect(url_for("cerca_giocatori"))

    return render_template("cerca_giocatori.html")

def calculate_similarity(input_text, dataframe, model):
    input_vector = text_to_vector(input_text, model)
    input_words = input_text.split()
    similarities = []

    for _, row in dataframe.iterrows():
        row_text = " ".join(str(value) for value in row.values if pd.notnull(value))
        row_words = row_text.split()
        row_vector = text_to_vector(row_text, model)
        vector_similarity = cosine_similarity(
            input_vector.reshape(1, -1), row_vector.reshape(1, -1)
        )[0][0]
        exact_match_score = sum(1 for word in input_words if word in row_words)
        combined_score = 0.9 * vector_similarity + 0.1 * exact_match_score
        similarities.append(combined_score)

    dataframe["similarity_score"] = similarities
    return dataframe.sort_values(by="similarity_score", ascending=False).head(10)

def text_to_vector(text, model):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, flash, session
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sqlite3
import re
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.secret_key = "replace_this_with_a_strong_secret"


ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
PARAPHRASE_MODEL = "t5-base"
DEVICE = 0 if torch.cuda.is_available() else -1

classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device=DEVICE)
par_tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
par_model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL).to("cuda" if DEVICE == 0 else "cpu")


DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def add_user(username, phone, email, password):
    pw_hash = generate_password_hash(password)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, phone, email, password_hash) VALUES (?, ?, ?, ?)",
                  (username, phone, email, pw_hash))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError as e:
        return False, str(e)
    finally:
        conn.close()

def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, phone, email, password_hash FROM users WHERE email = ?", (email,))
    row = c.fetchone()
    conn.close()
    return row


def detect_fake_or_original(text, threshold=0.60):
    candidate_labels = ["original", "fake"]
    res = classifier(text, candidate_labels)
    top_label = res["labels"][0]
    top_score = float(res["scores"][0])
    if top_score < threshold:
        return {"label": "uncertain", "score": top_score}
    return {"label": top_label, "score": top_score}

def paraphrase(text, max_length=128, num_beams=4, num_return_sequences=2):
    device = "cuda" if DEVICE == 0 else "cpu"
    prefix = "paraphrase: "
    input_text = prefix + text.strip().replace("\n", " ")
    inputs = par_tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)

    outputs = par_model.generate(
        inputs,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )

    out_texts = [
        par_tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for o in outputs
    ]
    return out_texts


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        phone = request.form.get("phone", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not phone or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("signup"))

        if not re.fullmatch(r"\d{10}", phone):
            flash("Phone number must be exactly 10 digits.", "error")
            return redirect(url_for("signup"))

        if not email.endswith("@gmail.com"):
            flash("Email must be a gmail address ending with @gmail.com", "error")
            return redirect(url_for("signup"))

        success, err = add_user(username, phone, email, password)
        if not success:
            flash(f"Sign up failed: {err}", "error")
            return redirect(url_for("signup"))

        flash("Sign up successful. Please sign in.", "success")
        return redirect(url_for("signin"))

    return render_template("signup.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = get_user_by_email(email)
        if not user:
            flash("User not found. Please sign up first.", "error")
            return redirect(url_for("signup"))

        user_id, username, phone, email_db, pw_hash = user

        if not check_password_hash(pw_hash, password):
            flash("Incorrect password.", "error")
            return redirect(url_for("signin"))

        session["user_id"] = user_id
        session["username"] = username

        flash("Signed in successfully.", "success")
        return redirect(url_for("home"))

    return render_template("signin.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))


@app.route("/home")
def home():
    if "user_id" not in session:
        return redirect(url_for("signin"))
    return render_template("home.html", username=session.get("username"))


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/description")
def description():
    return render_template("description.html")


@app.route("/fakenews", methods=["GET", "POST"])
def fakenews():
    if "user_id" not in session:
        return redirect(url_for("signin"))

    result = None
    paraphrases = []
    text = ""

    if request.method == "POST":
        text = request.form["text"]
        if text.strip():
            result = detect_fake_or_original(text)
            paraphrases = paraphrase(text)

    return render_template(
        "fakenews.html",
        result=result,
        text=text,
        paraphrases=paraphrases
    )


if __name__ == "__main__":
    app.run(debug=True)

import os
import re
import hmac
import secrets
import sqlite3
from pathlib import Path
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, g, abort
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from recommender import recommend_papers


BASE_DIR = Path(__file__).resolve().parent

INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(exist_ok=True)

DATABASE = INSTANCE_DIR / "research_app.db"

app = Flask(__name__)

app.config["SECRET_KEY"] = (
    os.environ.get("APP_SECRET_KEY")
    or secrets.token_hex(32)
)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = (
    os.environ.get("COOKIE_SECURE") == "1"
)


PAPERS = [
    {
        "id": 1,
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani et al.",
        "year": 2017,
        "abstract": "Introduces the Transformer architecture for natural language processing.",
        "url": "https://arxiv.org/abs/1706.03762"
    },
    {
        "id": 2,
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": "Jacob Devlin et al.",
        "year": 2018,
        "abstract": "A language representation model using bidirectional pre-training.",
        "url": "https://arxiv.org/abs/1810.04805"
    },
    {
        "id": 3,
        "title": "Deep Residual Learning for Image Recognition",
        "authors": "Kaiming He et al.",
        "year": 2015,
        "abstract": "Introduces residual neural networks for image recognition.",
        "url": "https://arxiv.org/abs/1512.03385"
    },
    {
        "id": 4,
        "title": "Generative Adversarial Networks",
        "authors": "Ian Goodfellow et al.",
        "year": 2014,
        "abstract": "Introduces adversarial training for generative models.",
        "url": "https://arxiv.org/abs/1406.2661"
    },
    {
        "id": 5,
        "title": "Deep Learning",
        "authors": "Yoshua Bengio, Ian Goodfellow, Aaron Courville",
        "year": 2015,
        "abstract": "An introduction to deep learning methods and applications.",
        "url": "https://arxiv.org/abs/1603.05691"
    },
    {
        "id": 6,
        "title": "Machine Learning: A Probabilistic Perspective",
        "authors": "Kevin P. Murphy",
        "year": 2012,
        "abstract": "Probabilistic methods and foundations of machine learning.",
        "url": "https://probml.github.io/pml-book/"
    },
    {
        "id": 7,
        "title": "Playing Atari with Deep Reinforcement Learning",
        "authors": "Volodymyr Mnih et al.",
        "year": 2013,
        "abstract": "Uses deep reinforcement learning to learn Atari games.",
        "url": "https://arxiv.org/abs/1312.5602"
    },
    {
        "id": 8,
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        "authors": "Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton",
        "year": 2012,
        "abstract": "Demonstrates deep convolutional networks for image classification.",
        "url": "https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html"
    },
    {
        "id": 9,
        "title": "Show and Tell: A Neural Image Caption Generator",
        "authors": "Oriol Vinyals et al.",
        "year": 2015,
        "abstract": "Combines computer vision and language generation to describe images.",
        "url": "https://arxiv.org/abs/1411.4555"
    },
    {
        "id": 10,
        "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "authors": "Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio",
        "year": 2014,
        "abstract": "Introduces attention for neural machine translation.",
        "url": "https://arxiv.org/abs/1409.0473"
    },
    {
        "id": 11,
        "title": "Language Models are Few-Shot Learners",
        "authors": "Tom B. Brown et al.",
        "year": 2020,
        "abstract": "Studies few-shot learning capabilities of large language models.",
        "url": "https://arxiv.org/abs/2005.14165"
    },
    {
        "id": 12,
        "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        "authors": "Alexey Dosovitskiy et al.",
        "year": 2020,
        "abstract": "Applies Transformer architectures to image classification.",
        "url": "https://arxiv.org/abs/2010.11929"
    },
    {
        "id": 13,
        "title": "Denoising Diffusion Probabilistic Models",
        "authors": "Jonathan Ho, Ajay Jain, Pieter Abbeel",
        "year": 2020,
        "abstract": "Presents diffusion models for generating data.",
        "url": "https://arxiv.org/abs/2006.11239"
    },
    {
        "id": 14,
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Patrick Lewis et al.",
        "year": 2020,
        "abstract": "Combines information retrieval with text generation.",
        "url": "https://arxiv.org/abs/2005.11401"
    },
    {
        "id": 15,
        "title": "Adam: A Method for Stochastic Optimization",
        "authors": "Diederik P. Kingma, Jimmy Ba",
        "year": 2014,
        "abstract": "Introduces the Adam optimization algorithm for training models.",
        "url": "https://arxiv.org/abs/1412.6980"
    },
    {
        "id": 16,
        "title": "The Llama 3 Herd of Models",
        "authors": "Aaron Grattafiori et al.",
        "year": 2024,
        "abstract": "Describes the Llama 3 family of large language models.",
        "url": "https://arxiv.org/abs/2407.21783"
    },
    {
        "id": 17,
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "authors": "Nils Reimers, Iryna Gurevych",
        "year": 2019,
        "abstract": "Creates sentence embeddings for semantic similarity and search.",
        "url": "https://arxiv.org/abs/1908.10084"
    },
    {
        "id": 18,
        "title": "Deep Neural Networks for YouTube Recommendations",
        "authors": "Paul Covington, Jay Adams, Emre Sargin",
        "year": 2016,
        "abstract": "Describes a deep learning system for video recommendations.",
        "url": "https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/"
    },
    {
        "id": 19,
        "title": "XGBoost: A Scalable Tree Boosting System",
        "authors": "Tianqi Chen, Carlos Guestrin",
        "year": 2016,
        "abstract": "Introduces a scalable tree boosting system for machine learning.",
        "url": "https://arxiv.org/abs/1603.02754"
    },
    {
        "id": 20,
        "title": "A Survey on Data Collection for Machine Learning",
        "authors": "Yuji Roh, Geon Heo, Steven Euijong Whang",
        "year": 2018,
        "abstract": "Reviews data collection, data labeling, and machine learning datasets.",
        "url": "https://arxiv.org/abs/1811.03402"
    }
]


def get_db():

    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row

    return g.db


@app.teardown_appcontext
def close_db(exception=None):

    db = g.pop("db", None)

    if db is not None:
        db.close()


def init_db():

    with sqlite3.connect(DATABASE) as db:

        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS saved_papers (
                user_id INTEGER NOT NULL,
                paper_id INTEGER NOT NULL,
                PRIMARY KEY (user_id, paper_id),
                FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
        """)

        db.execute("PRAGMA foreign_keys = ON")


def login_required(view):

    @wraps(view)
    def wrapped_view(*args, **kwargs):

        if "user_id" not in session:
            flash("Please log in first.")
            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped_view


@app.before_request
def csrf_protection():

    if request.method in ("POST", "PUT", "PATCH", "DELETE"):

        submitted = request.form.get("csrf_token")

        if submitted is None:
            data = request.get_json(silent=True) or {}
            submitted = data.get("csrf_token")

        expected = session.get("csrf_token", "")

        if (
            not isinstance(submitted, str)
            or not expected
            or not hmac.compare_digest(submitted, expected)
        ):
            abort(400, description="Invalid security token.")


@app.context_processor
def inject_template_values():

    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)

    return {"csrf_token": session["csrf_token"]}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():

    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not name or len(name) > 100:
            flash("Enter a valid name.")
            return render_template("register.html")

        if (
            not email
            or len(email) > 254
            or not re.fullmatch(
                r"[^@\s]+@[^@\s]+\.[^@\s]+",
                email
            )
        ):
            flash("Enter a valid email address.")
            return render_template("register.html")

        if len(password) < 8 or len(password) > 128:
            flash("Password must contain 8 to 128 characters.")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.")
            return render_template("register.html")

        password_hash = generate_password_hash(password)

        try:

            db = get_db()

            cursor = db.execute(
                """
                INSERT INTO users (name, email, password_hash)
                VALUES (?, ?, ?)
                """,
                (name, email, password_hash)
            )

            db.commit()

        except sqlite3.IntegrityError:

            flash("This email is already registered.")
            return render_template("register.html")

        session.clear()
        session["user_id"] = cursor.lastrowid
        session["user_name"] = name
        session["csrf_token"] = secrets.token_urlsafe(32)

        flash("Registration successful!")

        return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = get_db().execute(
            """
            SELECT id, name, email, password_hash
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        if user is None or not check_password_hash(
            user["password_hash"], password
        ):

            flash("Incorrect email or password.")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        session["csrf_token"] = secrets.token_urlsafe(32)

        flash("Login successful!")

        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():

    session.clear()

    flash("You have logged out successfully.")

    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():

    papers = recommend_papers(
        "artificial intelligence",
        PAPERS
    )[:3]

    return render_template(
        "dashboard.html",
        papers=papers,
        user_name=session.get("user_name", "User")
    )


@app.route("/search", methods=["GET", "POST"])
@login_required
def search():

    results = []
    query = ""

    if request.method == "POST":

        query = request.form.get("query", "").strip()

        if not query:
            flash("Please enter a research topic.")

        elif len(query) > 500:
            flash("Search query is too long.")

        else:
            results = recommend_papers(query, PAPERS)

            if not results:
                flash("No matching papers found.")

    return render_template(
        "search.html",
        results=results,
        query=query
    )


@app.route("/save/<int:paper_id>", methods=["POST"])
@login_required
def save_paper(paper_id):

    paper = next(
        (p for p in PAPERS if p["id"] == paper_id),
        None
    )

    if paper is None:
        abort(404)

    db = get_db()

    db.execute(
        """
        INSERT OR IGNORE INTO saved_papers (user_id, paper_id)
        VALUES (?, ?)
        """,
        (session["user_id"], paper_id)
    )

    db.commit()

    flash("Paper saved successfully.")

    return redirect(request.referrer or url_for("saved"))


@app.route("/saved")
@login_required
def saved():

    rows = get_db().execute(
        """
        SELECT paper_id
        FROM saved_papers
        WHERE user_id = ?
        """,
        (session["user_id"],)
    ).fetchall()

    saved_ids = {row["paper_id"] for row in rows}

    saved_papers = [
        paper for paper in PAPERS
        if paper["id"] in saved_ids
    ]

    return render_template(
        "saved.html",
        papers=saved_papers
    )


@app.route("/remove/<int:paper_id>", methods=["POST"])
@login_required
def remove_paper(paper_id):

    db = get_db()

    db.execute(
        """
        DELETE FROM saved_papers
        WHERE user_id = ? AND paper_id = ?
        """,
        (session["user_id"], paper_id)
    )

    db.commit()

    flash("Paper removed successfully.")

    return redirect(url_for("saved"))


with app.app_context():
    init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

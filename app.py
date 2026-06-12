from flask import Flask, render_template, request, redirect, url_for, session, flash
from functools import wraps
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from db import get_connection
from preprocessing import preprocess
from preprocess_data import update_preprocessing

from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "sistemrekomendasi_bukuremaja"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'login' not in session:
            flash('Silakan login terlebih dahulu.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def load_data():
    conn = get_connection()
    query = """
    SELECT
        buku.ID_Buku,
        buku.Judul,
        buku.Kategori,
        buku.Penulis,
        buku.Penerbit,
        buku.Deskripsi,
        buku.Gambar,
        buku_processed.teks_bersih
    FROM buku
    JOIN buku_processed
    ON buku.ID_Buku = buku_processed.ID_Buku
    """
    data = pd.read_sql(query, conn)
    conn.close()
    data = data.fillna("")
    return data



def get_recommendations(query, n=10):
    data = load_data()

    vectorizer = CountVectorizer()

    term_matrix = vectorizer.fit_transform(
        data["teks_bersih"]
    ).toarray()

    terms = vectorizer.get_feature_names_out()

    N = len(data)

    # DF
    df_values = np.count_nonzero(
        term_matrix > 0,
        axis=0
    )

    # PREPROCESS QUERY
    query_asli = query
    query = preprocess(query)
    query_counts = vectorizer.transform([query]).toarray().astype(float)
    query_word_indices = np.where(
        query_counts[0] > 0
    )[0]

    if len(query_word_indices) == 0:
        return []

    # TF
    total_query = query_counts.sum()
    tf_query = query_counts / total_query

    tf = np.zeros_like(
        term_matrix,
        dtype=float
    )
    for i in range(len(term_matrix)):

        total_term_query = np.count_nonzero(
            term_matrix[i][query_word_indices]
        )

        if total_term_query > 0:
            tf[i, query_word_indices
            ] = ( term_matrix[i][query_word_indices] /total_term_query)

    # IDF
    idf = np.zeros(len(terms))
    idf[
        query_word_indices
    ] = np.where(

        df_values[
            query_word_indices
        ] > 0,

        np.log10(N / df_values[ query_word_indices ] ), 0
    )

    # TF-IDF
    tfidf_query = tf_query * idf
    tfidf = tf * idf

    # COSINE SIMILARITY
    similarity = cosine_similarity(
        tfidf_query,
        tfidf
    ).flatten()

    # SIMPAN RIWAYAT
    tfidf_text = ""
    for idx in query_word_indices:

        tfidf_text += (
            f"{terms[idx]} = "
            f"{round(tfidf_query[0][idx],3)}\n"
        )

    top_similarity = np.argsort(similarity)[::-1][:10]

    similarity_text = ""
    ranking = 1
    for idx in top_similarity:
        if similarity[idx] > 0:
            similarity_text += (
                f"{ranking}. "
                f"{data.iloc[idx]['Judul']} "
                f"({round(similarity[idx],3)})\n"
            )
            ranking += 1
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO riwayat_perhitungan
        (
            Query_Pencarian,
            Hasil_TFIDF,
            Hasil_Similarity
        )
        VALUES (%s,%s,%s)
    """,
    (
        query_asli,
        tfidf_text,
        similarity_text
    ))

    conn.commit()
    conn.close()

    # HASIL REKOMENDASI
    top_indices = similarity.argsort()[::-1]

    hasil = data.iloc[
        top_indices
    ].copy()

    hasil[
        "similarity_score"
    ] = similarity[
        top_indices
    ]

    hasil = hasil[
        hasil[
            "similarity_score"
        ] > 0
    ]

    hasil = hasil.drop_duplicates(
        subset="Judul"
    )

    hasil = hasil.head(n)

    return hasil.to_dict(
        orient="records"
    )

# DETAIL BUKU
def get_book_by_id(book_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM buku WHERE ID_Buku=%s", (book_id,))
    book = cursor.fetchone()
    cursor.close()
    conn.close()
    return book

# =========================
# ROUTING
# =========================
@app.route("/")
def index():
    return render_template("beranda.html")

@app.route("/cari_rekomendasi")
def rekomendasi():
    return render_template("rekomendasi.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    query = request.form.get("title")
    hasil = get_recommendations(query)
    return render_template(
        "hasil.html",
        title=query,
        hasil=hasil
    )

@app.route("/book/<int:book_id>")
def book_detail(book_id):
    book = get_book_by_id(book_id)
    if not book:
        return "Buku tidak ditemukan", 404
    return render_template("detail.html", book=book)

@app.route('/buku')
def buku():
    page = request.args.get('page', 1, type=int)
    keyword = request.args.get('keyword', '').strip()
    selected_kategori = request.args.get('kategori', '').strip()
    per_page = 12
    offset = (page - 1) * per_page

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT ID_Kategori, Kategori 
        FROM kategori_buku 
        ORDER BY Kategori ASC
    """)
    kategori_list = cursor.fetchall()

    conditions = []
    params = []

    if keyword:
        search = f"%{keyword}%"
        conditions.append("""
            (b.Judul LIKE %s
            OR b.Penulis LIKE %s
            OR b.Penerbit LIKE %s
            OR k.Kategori LIKE %s
            OR CAST(b.Tahun_Terbit AS CHAR) LIKE %s)
        """)
        params.extend([search, search, search, search, search])

    if selected_kategori:
        conditions.append("k.Kategori = %s")
        params.append(selected_kategori)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    cursor.execute(f"""
        SELECT COUNT(*) AS total
        FROM buku b
        LEFT JOIN kategori_buku k ON b.ID_Kategori = k.ID_Kategori
        {where_clause}
    """, params)
    total = cursor.fetchone()['total']

    cursor.execute(f"""
        SELECT b.*, k.Kategori
        FROM buku b
        LEFT JOIN kategori_buku k ON b.ID_Kategori = k.ID_Kategori
        {where_clause}
        LIMIT %s OFFSET %s
    """, params + [per_page, offset])

    results = cursor.fetchall()
    total_pages = (total + per_page - 1) // per_page

    cursor.close()
    conn.close()

    return render_template(
        'buku.html',
        results=results,
        page=page,
        total_pages=total_pages,
        keyword=keyword,
        selected_kategori=selected_kategori,
        kategori_list=kategori_list
    )


@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        nama_lengkap = request.form['nama_lengkap']
        email = request.form['email']
        password = request.form['password']
        password_confirm = request.form['password_confirm']

        # Validasi konfirmasi password
        if password != password_confirm:
            flash('Konfirmasi password tidak sesuai', 'warning')
            return redirect(url_for('register'))

        conn = get_connection()
        cursor = conn.cursor()

        # Cek email
        cursor.execute(
            """
            SELECT *
            FROM tbl_admin
            WHERE Email = %s
            """,
            (email,)
        )

        cek = cursor.fetchone()

        if cek:
            cursor.close()
            conn.close()

            flash('Email sudah terdaftar', 'danger')
            return redirect(url_for('register'))

        password_hash = generate_password_hash(password)

        cursor.execute("""
            INSERT INTO tbl_admin
            (
                Nama_Lengkap,
                Email,
                Password
            )
            VALUES
            (
                %s,
                %s,
                %s
            )
        """,
        (nama_lengkap, email, password_hash ))

        conn.commit()

        cursor.close()
        conn.close()

        flash('Registrasi berhasil, silakan login', 'success')

        return redirect(url_for('login'))

    return render_template('admin/register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT *
            FROM tbl_admin
            WHERE Email = %s
        """, (email,))

        admin = cursor.fetchone()
        conn.close()

        if admin and check_password_hash(admin['Password'], password):
            session['login'] = True
            session['id_admin'] = admin['ID_Admin']
            session['nama_admin'] = admin['Nama_Lengkap']
            return redirect(url_for('dashboard'))

        flash('Email atau Password Salah')

    return render_template('admin/login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT COUNT(*) AS total_admin FROM tbl_admin")
    total_admin = cursor.fetchone()['total_admin']

    cursor.execute("SELECT COUNT(*) AS jumlah_buku FROM buku")
    jumlah_buku = cursor.fetchone()['jumlah_buku']

    cursor.execute("SELECT COUNT(*) AS total_kategori FROM kategori_buku")
    total_kategori = cursor.fetchone()['total_kategori']

    cursor.execute("""
        SELECT COUNT(DISTINCT Penerbit) AS total_penerbit
        FROM buku
    """)
    total_penerbit = cursor.fetchone()['total_penerbit']

    cursor.close()
    conn.close()

    return render_template(
        'admin/dashboard.html',
        total_admin=total_admin,
        jumlah_buku=jumlah_buku,
        total_kategori=total_kategori,
        total_penerbit=total_penerbit
    )

@app.route('/data_buku')
@login_required
def data_buku():
    page = request.args.get('page', 1, type=int)
    per_page = 10

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT COUNT(*) as total FROM buku")
    total_data = cursor.fetchone()['total']

    offset = (page - 1) * per_page

    cursor.execute("""
        SELECT *
        FROM buku
        LIMIT %s OFFSET %s
    """, (per_page, offset))
    hasil = cursor.fetchall()

    cursor.execute("SELECT * FROM kategori_buku")
    kategori = cursor.fetchall()

    total_pages = (total_data + per_page - 1) // per_page

    cursor.close()
    conn.close()

    return render_template(
        'admin/data_buku.html',
        hasil=hasil,
        kategori=kategori,
        page=page,
        per_page=per_page,
        total_pages=total_pages
    )

@app.route('/tambah_buku', methods=['POST'])
@login_required
def tambah_buku():
    id_buku = int(request.form.get('id_buku'))

    judul = request.form.get('judul')
    id_kategori = request.form.get('id_kategori')
    penulis = request.form.get('penulis')
    penerbit = request.form.get('penerbit')
    tahun_terbit = request.form.get('tahun_terbit')
    deskripsi = request.form.get('deskripsi')
    gambar_type = request.form.get('gambar_type')

    gambar = ""

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT ID_Buku FROM buku WHERE ID_Buku = %s",
        (id_buku,)
    )

    cek = cursor.fetchone()

    if cek:
        cursor.close()
        conn.close()

        flash('ID Buku sudah digunakan!', 'danger')
        return redirect(url_for('data_buku'))

    cursor.execute("""
        SELECT Kategori
        FROM kategori_buku
        WHERE ID_Kategori = %s
    """, (id_kategori,))

    hasil = cursor.fetchone()
    kategori = hasil['Kategori'] if hasil else ""

    # PROSES GAMBAR
    if gambar_type == "url":

        gambar = request.form.get(
            'gambar_url',
            ''
        ).strip()

    elif gambar_type == "upload":

        gambar_file = request.files.get('gambar_file')

        if gambar_file and gambar_file.filename != "":

            filename = secure_filename(gambar_file.filename)
            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                filename
            )

            gambar_file.save(filepath)
            gambar = "uploads/" + filename

    cursor.execute("""
        INSERT INTO buku
        (
            ID_Buku,
            Judul,
            Kategori,
            ID_Kategori,
            Penulis,
            Penerbit,
            Tahun_Terbit,
            Deskripsi,
            Gambar
        )
        VALUES
        (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        id_buku,
        judul,
        kategori,
        id_kategori,
        penulis,
        penerbit,
        tahun_terbit,
        deskripsi,
        gambar
    ))

    conn.commit()
    cursor.close()
    conn.close()

    update_preprocessing(id_buku)
    flash('Buku berhasil ditambahkan.','success')
    return redirect(url_for('data_buku')
    )

@app.route('/edit_buku/<int:id_buku>', methods=['POST'])
@login_required
def edit_buku(id_buku):

    judul = request.form.get('judul')
    id_kategori = request.form.get('id_kategori')
    penulis = request.form.get('penulis')
    penerbit = request.form.get('penerbit')
    tahun_terbit = request.form.get('tahun_terbit')
    deskripsi = request.form.get('deskripsi')
    gambar_type = request.form.get('gambar_type')

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT Kategori
        FROM kategori_buku
        WHERE ID_Kategori = %s
    """, (id_kategori,))

    hasil = cursor.fetchone()
    kategori = hasil['Kategori'] if hasil else ""

    cursor.execute("""
        SELECT Gambar
        FROM buku
        WHERE ID_Buku = %s
    """, (id_buku,))

    buku_lama = cursor.fetchone()

    gambar = ""
    if buku_lama:
        gambar = buku_lama['Gambar']

    if gambar_type == "url":

        gambar_url = request.form.get(
            'gambar_url',
            ''
        ).strip()

        if gambar_url:
            gambar = gambar_url

    # Upload gambar
    elif gambar_type == "upload":

        gambar_file = request.files.get('gambar_file')

        if gambar_file and gambar_file.filename != "":
            filename = secure_filename(
                gambar_file.filename)
            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                filename)
            
            gambar_file.save(filepath)
            gambar = "uploads/" + filename

    # Update data buku
    cursor.execute("""
        UPDATE buku
        SET
            Judul=%s,
            Kategori=%s,
            ID_Kategori=%s,
            Penulis=%s,
            Penerbit=%s,
            Tahun_Terbit=%s,
            Deskripsi=%s,
            Gambar=%s
        WHERE ID_Buku=%s
    """, (
        judul,
        kategori,
        id_kategori,
        penulis,
        penerbit,
        tahun_terbit,
        deskripsi,
        gambar,
        id_buku
    ))

    conn.commit()
    cursor.close()
    conn.close()

    update_preprocessing(id_buku)
    flash('Data buku berhasil diperbarui.', 'success')
    return redirect(url_for('data_buku'))

@app.route('/hapus_buku/<int:book_id>')
@login_required
def hapus_buku(book_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM buku WHERE ID_Buku=%s", (book_id,))
    conn.commit()
    cursor.close()
    conn.close()

    flash('Buku berhasil dihapus.', 'success')
    return redirect(url_for('data_buku'))

@app.route('/detail_buku/<int:book_id>')
@login_required
def detail_buku(book_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM buku WHERE ID_Buku=%s", (book_id,))
    book = cursor.fetchone()
    cursor.close()
    conn.close()
    return render_template('admin/detail_buku.html', book=book)

@app.route('/kategori_buku')
@login_required
def kategori_buku():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM kategori_buku ORDER BY ID_Kategori ASC")
    kategori = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('admin/kategori_buku.html', kategori=kategori)

@app.route('/tambah_kategori', methods=['POST'])
@login_required
def tambah_kategori():

    id_kategori = int(request.form['ID_Kategori'])
    kategori = request.form['kategori'].strip()

    if not kategori:
        flash('Nama kategori tidak boleh kosong!', 'danger')
        return redirect(url_for('kategori_buku'))

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Cek ID kategori
    cursor.execute("""
        SELECT *
        FROM kategori_buku
        WHERE ID_Kategori = %s
    """, (id_kategori,))

    cek_id = cursor.fetchone()

    if cek_id:
        cursor.close()
        conn.close()

        flash('ID Kategori sudah digunakan!', 'danger')
        return redirect(url_for('kategori_buku'))

    # Cek nama kategori
    cursor.execute("""
        SELECT *
        FROM kategori_buku
        WHERE LOWER(Kategori) = LOWER(%s)
    """, (kategori,))

    cek_nama = cursor.fetchone()

    if cek_nama:
        cursor.close()
        conn.close()

        flash('Kategori sudah ada!', 'danger')
        return redirect(url_for('kategori_buku'))

    cursor.close()

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO kategori_buku
        (
            ID_Kategori,
            Kategori
        )
        VALUES
        (%s, %s)
    """, (
        id_kategori,
        kategori
    ))

    conn.commit()
    cursor.close()
    conn.close()

    flash('Kategori berhasil ditambahkan.', 'success')
    return redirect(url_for('kategori_buku'))

@app.route('/edit_kategori/<int:id>', methods=['POST'])
@login_required
def edit_kategori(id):
    kategori = request.form['kategori'].strip()

    if not kategori:
        flash('Nama kategori tidak boleh kosong!')
        return redirect(url_for('kategori_buku'))

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT * FROM kategori_buku
        WHERE LOWER(Kategori)=LOWER(%s)
        AND ID_Kategori != %s
    """, (kategori, id))
    cek = cursor.fetchone()

    if cek:
        flash('Nama kategori sudah digunakan!')
        cursor.close()
        conn.close()
        return redirect(url_for('kategori_buku'))

    cursor.close()
    cursor = conn.cursor()
    cursor.execute("UPDATE kategori_buku SET Kategori=%s WHERE ID_Kategori=%s", (kategori, id))
    conn.commit()
    cursor.close()
    conn.close()
    
    flash('Kategori berhasil diperbarui.')
    return redirect(url_for('kategori_buku'))

@app.route('/hapus_kategori/<int:id>')
@login_required
def hapus_kategori(id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT COUNT(*) AS total FROM buku WHERE ID_Kategori=%s
    """, (id,))
    cek = cursor.fetchone()

    if cek['total'] > 0:
        flash('Kategori tidak dapat dihapus karena masih digunakan oleh data buku.')
        cursor.close()
        conn.close()
        return redirect(url_for('kategori_buku'))

    cursor.close()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM kategori_buku WHERE ID_Kategori=%s", (id,))
    conn.commit()
    cursor.close()
    conn.close()
    
    flash('Kategori berhasil dihapus.')
    return redirect(url_for('kategori_buku'))

@app.route('/daftar_admin')
@login_required
def daftar_admin():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT ID_Admin, Nama_Lengkap, Email
        FROM tbl_admin
        ORDER BY ID_Admin ASC
    """)
    admin = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('admin/daftar_admin.html', admin=admin)

@app.route('/riwayat_perhitungan')
@login_required
def riwayat_perhitungan():

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT *
        FROM riwayat_perhitungan
        ORDER BY ID_Riwayat DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return render_template(
        'admin/riwayat_perhitungan.html',
        data=data
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
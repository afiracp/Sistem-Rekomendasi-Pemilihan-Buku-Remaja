import pandas as pd
from db import get_connection
from preprocessing import case_folding, tokenizing, filtering, stemming


def update_preprocessing(id_buku=None):

    conn = get_connection()

    if id_buku is None:

        data = pd.read_sql(
            "SELECT * FROM buku",
            conn
        )

    else:

        data = pd.read_sql(
            """
            SELECT *
            FROM buku
            WHERE ID_Buku=%s
            """,
            conn,
            params=(id_buku,)
        )

    conn.close()

    if data.empty:
        return

    data = data.fillna("")

    data["Fitur_Gabungan"] = (
        data["Judul"].astype(str) + " " +
        data["Penulis"].astype(str) + " " +
        data["Kategori"].astype(str) + " " +
        data["Penerbit"].astype(str) + " " +
        data["Deskripsi"].astype(str)
    )

    for i, row in data.iterrows():

        cf = case_folding(row["Fitur_Gabungan"])
        tk = tokenizing(cf)
        fl = filtering(tk)
        st = stemming(fl)

        data.at[i, "Case_Folding"] = cf
        data.at[i, "Tokenizing"] = " ".join(tk)
        data.at[i, "Filtering"] = " ".join(fl)
        data.at[i, "Stemming"] = " ".join(st)
        data.at[i, "Teks_Bersih"] = " ".join(st)

    conn = get_connection()
    cursor = conn.cursor()

    if id_buku is None:

        cursor.execute(
            "DELETE FROM buku_processed"
        )

    for _, row in data.iterrows():

        cursor.execute("""
            DELETE FROM buku_processed
            WHERE ID_Buku=%s
        """, (
            row["ID_Buku"],
        ))

        cursor.execute("""
            INSERT INTO buku_processed
            (
                ID_Buku,
                Fitur_Gabungan,
                Case_Folding,
                Tokenizing,
                Filtering,
                Stemming,
                Teks_Bersih
            )
            VALUES
            (%s,%s,%s,%s,%s,%s,%s)
        """, (
            row["ID_Buku"],
            row["Fitur_Gabungan"],
            row["Case_Folding"],
            row["Tokenizing"],
            row["Filtering"],
            row["Stemming"],
            row["Teks_Bersih"]
        ))

    conn.commit()

    cursor.close()
    conn.close()
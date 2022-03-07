"""A script to interact with smol sql database"""

import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    return sqlite3.connect(db_file)

db = create_connection('data_management/data/sources.db')


def generate_piece_table(pieces):
    cursor = db.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pieces (
        id integer PRIMARY KEY,
        author text,
        translator text,
        audience text,
        performance_date text,
        performance_type text,
        performance_work text,
        performance_location text,
        performance_performers text,
        full_text text,
        excerpt text,
        key_words text,
        notes text,
        context text,
        citation text,
        date text,
        type text,
        language text
    );''')
    for p in pieces:
        print('insertion time')
        sql = '''INSERT INTO pieces(author, translator, audience,
                    performance_date, performance_type, performance_work, performance_location, performance_performers,
                    full_text, excerpt, key_words, notes,
                    context, citation, date, type, language)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);'''

        cursor.execute(sql, (p['author'], p['translator'], p['audience'],
            p['performance_date'], p['performance_type'], p['performance_work'], p['performance_location'], p['performance_performers'],
            p['full_text'], p['excerpt'], p['key_words'], p['notes'],
            p['context'], p['citation'], p['date'], p['type'], p['language']))
    db.commit()   
    

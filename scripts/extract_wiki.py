import sqlite3
import os
import re

wiki_db_path = r"D:\soso\projects\Loong-pearl\data\raw\zhwiki.db"
corpus_path = r"D:\soso\projects\Loong-pearl\data\corpus.txt"

print("处理维基百科...")

conn = sqlite3.connect(wiki_db_path)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM articles")
total = cursor.fetchone()[0]
print(f"文章总数: {total}")

batch_size = 10000
max_articles = 100000

with open(corpus_path, 'a', encoding='utf-8') as f:
    count = 0
    written = 0
    
    cursor.execute("SELECT title, text FROM articles WHERE text IS NOT NULL")
    
    for title, text in cursor:
        if text and len(text) > 50:
            sentences = re.split(r'[。！？\n]', text)
            for sentence in sentences:
                sentence = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', '', sentence)
                sentence = sentence.strip()
                if 10 < len(sentence) < 200:
                    f.write(sentence + '\n')
                    written += 1
        
        count += 1
        if count % batch_size == 0:
            print(f"  处理 {count}/{total}, 写入 {written} 句...")
        
        if count >= max_articles:
            break

conn.close()

total_lines = sum(1 for _ in open(corpus_path, 'r', encoding='utf-8'))
print(f"\n语料库总行数: {total_lines}")
print(f"文件大小: {os.path.getsize(corpus_path) / 1024 / 1024:.2f} MB")

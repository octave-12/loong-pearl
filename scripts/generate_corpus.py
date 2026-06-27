import json
import sqlite3
import os
import re

output_dir = r"D:\soso\projects\Loong-pearl\data"
raw_dir = os.path.join(output_dir, "raw")
os.makedirs(raw_dir, exist_ok=True)

corpus_lines = set()

print("=" * 60)
print("从知识源生成语料库")
print("=" * 60)

# 1. 四字词
print("\n[1/5] 处理四字词...")
four_char_path = os.path.join(raw_dir, "four_char_words.txt")
if os.path.exists(four_char_path):
    with open(four_char_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 4 and line.isalpha():
                corpus_lines.add(line)
    print(f"  四字词: {len(corpus_lines)} 条")

# 2. 成语
print("\n[2/5] 处理成语...")
idioms_path = os.path.join(raw_dir, "idioms.json")
if os.path.exists(idioms_path):
    with open(idioms_path, 'r', encoding='utf-8') as f:
        idioms = json.load(f)
    for idiom in idioms:
        if isinstance(idiom, str) and len(idiom) >= 4:
            corpus_lines.add(idiom)
    print(f"  累计: {len(corpus_lines)} 条")

# 3. 中英词典词条
print("\n[3/5] 处理中英词典...")
cedict_path = os.path.join(raw_dir, "cedict_parsed.json")
if os.path.exists(cedict_path):
    with open(cedict_path, 'r', encoding='utf-8') as f:
        cedict = json.load(f)
    count = 0
    for word, info in cedict.items():
        if isinstance(word, str) and len(word) >= 2:
            if not re.match(r'^[0-9\-\.\s]+$', word):
                if re.match(r'^[\u4e00-\u9fff]+', word):
                    corpus_lines.add(word)
                    if 'definitions' in info:
                        for definition in info['definitions'][:2]:
                            if len(definition) > 10:
                                corpus_lines.add(definition)
        count += 1
        if count % 100000 == 0:
            print(f"  处理 {count} 条...")
    print(f"  累计: {len(corpus_lines)} 条")

# 4. 汉字字典定义
print("\n[4/5] 处理汉字字典...")
unihan_path = os.path.join(raw_dir, "dict_unihan.json")
if os.path.exists(unihan_path):
    with open(unihan_path, 'r', encoding='utf-8') as f:
        unihan = json.load(f)
    for char, info in unihan.items():
        if isinstance(char, str) and len(char) == 1:
            if 'definition' in info and info['definition']:
                definitions = info['definition'].split(';')
                for definition in definitions:
                    definition = definition.strip()
                    if len(definition) > 5:
                        corpus_lines.add(definition)
    print(f"  累计: {len(corpus_lines)} 条")

# 5. 维基百科摘要
print("\n[5/5] 处理维基百科...")
wiki_db_path = os.path.join(raw_dir, "zhwiki.db")
if os.path.exists(wiki_db_path):
    try:
        conn = sqlite3.connect(wiki_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"  数据库表: {tables}")
        
        if 'pages' in tables:
            cursor.execute("SELECT COUNT(*) FROM pages")
            total = cursor.fetchone()[0]
            print(f"  维基页面数: {total}")
            
            cursor.execute("SELECT title, extract FROM pages WHERE extract IS NOT NULL AND LENGTH(extract) > 50")
            count = 0
            for title, extract in cursor:
                if extract and len(extract) > 50:
                    sentences = re.split(r'[。！？\n]', extract)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if 10 < len(sentence) < 200:
                            corpus_lines.add(sentence)
                count += 1
                if count % 10000 == 0:
                    print(f"  处理 {count}/{total} 页...")
                if count >= 100000:
                    break
        
        conn.close()
    except Exception as e:
        print(f"  维基百科处理失败: {e}")
    print(f"  累计: {len(corpus_lines)} 条")

# 写入语料库
print("\n" + "=" * 60)
print("写入语料库...")
corpus_path = os.path.join(output_dir, "corpus.txt")
with open(corpus_path, 'w', encoding='utf-8') as f:
    for line in sorted(corpus_lines):
        f.write(line + '\n')

print(f"语料库已生成: {corpus_path}")
print(f"总行数: {len(corpus_lines)}")
print(f"文件大小: {os.path.getsize(corpus_path) / 1024 / 1024:.2f} MB")
print("=" * 60)
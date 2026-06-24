"""测试增量PMI计算内存使用"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
import numpy as np

def compute_pmi_batch(corpus_chunk, window_size=5, min_count=2):
    char_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    total_chars = 0
    total_pairs = 0
    
    for text in corpus_chunk:
        chars = list(text)
        total_chars += len(chars)
        for char in chars:
            char_counts[char] += 1
        for i in range(len(chars)):
            for j in range(i + 1, min(i + window_size, len(chars))):
                pair = (chars[i], chars[j])
                pair_counts[pair] += 1
                total_pairs += 1
    
    return char_counts, pair_counts, total_chars, total_pairs

def compute_pmi_incremental(corpus_path, batch_size=2000, max_lines=5000, 
                           window_size=5, min_count=3, pmi_threshold=1.0,
                           max_pairs=10000):
    print(f'增量PMI计算 (batch_size={batch_size}, max_lines={max_lines}, max_pairs={max_pairs})')
    
    global_char_counts = defaultdict(int)
    global_pair_counts = defaultdict(int)
    global_total_chars = 0
    global_total_pairs = 0
    
    batch = []
    total_read = 0
    batch_num = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            batch.append(line)
            total_read += 1
            
            if len(batch) >= batch_size:
                batch_num += 1
                t0 = time.time()
                cc, pc, tc, tp = compute_pmi_batch(batch, window_size, min_count)
                
                for char, count in cc.items():
                    global_char_counts[char] += count
                for pair, count in pc.items():
                    global_pair_counts[pair] += count
                global_total_chars += tc
                global_total_pairs += tp
                
                del cc, pc, batch
                gc.collect()
                
                if len(global_pair_counts) > max_pairs:
                    sorted_pairs = sorted(global_pair_counts.items(), key=lambda x: x[1], reverse=True)
                    global_pair_counts = defaultdict(int, sorted_pairs[:max_pairs])
                    del sorted_pairs
                    gc.collect()
                    print(f'  Batch {batch_num}: {total_read} lines, '
                          f'{len(global_char_counts)} chars, {len(global_pair_counts)} pairs (trimmed) '
                          f'({time.time()-t0:.1f}s)')
                else:
                    print(f'  Batch {batch_num}: {total_read} lines, '
                          f'{len(global_char_counts)} chars, {len(global_pair_counts)} pairs '
                          f'({time.time()-t0:.1f}s)')
                batch = []
            
            if total_read >= max_lines:
                break
    
    if batch:
        cc, pc, tc, tp = compute_pmi_batch(batch, window_size, min_count)
        for char, count in cc.items():
            global_char_counts[char] += count
        for pair, count in pc.items():
            global_pair_counts[pair] += count
        global_total_chars += tc
        global_total_pairs += tp
        del cc, pc, batch
        gc.collect()
    
    print(f'计算PMI值...')
    pmi_pairs = []
    for (char_a, char_b), count in global_pair_counts.items():
        if count < min_count:
            continue
        p_a = global_char_counts[char_a] / global_total_chars
        p_b = global_char_counts[char_b] / global_total_chars
        p_ab = count / global_total_pairs
        pmi = np.log(p_ab / (p_a * p_b + 1e-10) + 1e-10)
        if pmi > pmi_threshold:
            pmi_pairs.append((char_a, char_b, float(pmi)))
    
    del global_char_counts, global_pair_counts
    gc.collect()
    
    print(f'PMI pairs: {len(pmi_pairs)}')
    return pmi_pairs

if __name__ == '__main__':
    t0 = time.time()
    pmi_pairs = compute_pmi_incremental(
        'data/corpus.txt',
        batch_size=1000,
        max_lines=5000,
        window_size=5,
        min_count=3,
        pmi_threshold=1.0,
        max_pairs=10000
    )
    print(f'\n完成! 用时: {time.time()-t0:.1f}s')
    print(f'示例PMI pairs: {pmi_pairs[:5]}')
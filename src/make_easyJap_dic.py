import pandas as pd
import MeCab

def make_easyJap_dic(path):
    # データ読み込み
    df = pd.read_excel(path)
    # iteratorに渡すarticlesの初期化
    articles = []
    # MeCabの準備
    mecabTagger = MeCab.Tagger("-Ochasen")

    # 辞書作成
    vocab = {}
    id2wd = {}
    la = len(df)
    for index, row in df.iterrows():
        hard = row['#日本語(原文)']
        easy = row['#やさしい日本語']
        eng = row['#英語(原文)']
        articles.append([hard, easy])

        node = mecabTagger.parseToNode(hard)
        node = node.next
        while node:
            word = node.surface
            if word not in vocab:
                wid = len(vocab)
                vocab[word] = wid
                id2wd[wid] = word
            node = node.next

        # abstract
        node = mecabTagger.parseToNode(easy)
        node = node.next
        while node:
            word = node.surface
            if word not in vocab:
                wid = len(vocab)
                vocab[word] = wid
                id2wd[wid] = word
            node = node.next

    wid = len(vocab)
    vocab["<EOS>"] = wid
    id2wd[wid] = "<EOS>"
    vocab_size = len(vocab)
    return mecabTagger, articles, vocab, id2wd, la, vocab_size
 

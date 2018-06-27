import json
import MeCab

def make_narou_dic(path):
    articles = json.load(open(path, "r"))
    mecabTagger = MeCab.Tagger("-Ochasen")

    # 辞書作成
    vocab = {}
    id2wd = {}
    la = len(articles)
    print("len_articles : {0}".format(la))
    for i in range(la):
        title = articles[i]["title"]
        abstract = articles[i]["abstract"]
        genre = articles[i]["genre"]
        tags = articles[i]["tags"]

        # title
        node = mecabTagger.parseToNode(title)
        node = node.next
        while node:
            word = node.surface
            if word not in vocab:
                wid = len(vocab)
                vocab[word] = wid
                id2wd[wid] = word
            node = node.next

        # abstract
        node = mecabTagger.parseToNode(abstract)
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
 

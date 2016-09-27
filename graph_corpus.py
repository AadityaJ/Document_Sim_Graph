import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from collections import defaultdict
from gensim import corpora, models, similarities
def find_sims(doc):
    ## convert to dictionary
    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1] for text in texts]

    from pprint import pprint  # pretty-printer
    pprint(texts)
    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
    print(dictionary)
    print(dictionary.token2id)

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
    for c in corpus:
        print(c)
    dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
    corpus = corpora.MmCorpus('/tmp/deerwester.mm') # comes from the first tutorial, "From strings to vectors"
    print(corpus)
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    vec_bow = dictionary.doc2bow(doc.lower().split())
    vec_lsi = lsi[vec_bow] # convert the query to LSI space
    index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it
    index.save('/tmp/deerwester.index')
    index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
    sims = index[vec_lsi] # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sims) # print sorted (document number, similarity score) 2-tuples
    return sims
def draw_graph(edgefrom):
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt

    G = nx.Graph()
    G.add_edges_from(edgefrom)
    nx.draw(G,with_labels=True,node_size=200,edge_vmin=1000,node_color='yellow')
    plt.show()
def getCorp(documents):
    #toDict()
    dictio=[]
    for txt in (documents):
        dictio.append(find_sims(txt))
    edgefrom=[]
    ix=0
    jx=0
    for ix in range(0,len(dictio)):
        for jx in range(0,len(dictio)):
            if dictio[ix][jx][1]>=0.5 and ix!=jx:
                m=ix,dictio[ix][jx][0]
                edgefrom.append(m)
    draw_graph(edgefrom)

f=open("ram.txt","r")
txt=f.read()
documents=txt.split('\n\n')

getCorp(documents)

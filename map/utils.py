import PyPDF2
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import os
import fitz
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from thefuzz import fuzz
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class Matcher:
    def __init__(self, cache_dir):
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.spacy = spacy.load("en_core_web_lg")
        self.sent_emb = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)

    def bert_matcher(self, sentences1, sentences2):
        # Tokenize input sentence and convert to tensor
        inputs1 = self.tokenizer(sentences1, return_tensors="pt", truncation=True, padding=True)
        inputs2 = self.tokenizer(sentences2, return_tensors="pt", truncation=True, padding=True)

        # Extract embeddings
        with torch.no_grad():
            outputs1 = self.model(**inputs1)
            outputs2 = self.model(**inputs2)

        # Use mean of the last hidden state as the sentence embedding
        embeddings1 = outputs1.last_hidden_state.mean(dim=1)
        embeddings2 = outputs2.last_hidden_state.mean(dim=1)

        # Normalize each embedding to have norm=1 to simplify cosine similarity computation
        normed_emb1 = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
        normed_emb2 = embeddings2 / embeddings2.norm(dim=1, keepdim=True)

        # Compute cosine similarities
        cos_sim_matrix = torch.mm(normed_emb1, normed_emb2.transpose(0, 1))

        seq = torch.argmax(cos_sim_matrix, dim=1).tolist()

        return seq

    def miniLM_matcher(self, sentences1, sentences2):
        # Tokenize input sentence and convert to tensor

        embeddings1 = torch.tensor(self.sent_emb.encode(sentences1))
        embeddings2 = torch.tensor(self.sent_emb.encode(sentences2))

        # import pdb; pdb.set_trace()

        # Normalize each embedding to have norm=1 to simplify cosine similarity computation
        normed_emb1 = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
        normed_emb2 = embeddings2 / embeddings2.norm(dim=1, keepdim=True)

        # Compute cosine similarities
        cos_sim_matrix = torch.mm(normed_emb1, normed_emb2.transpose(0, 1))

        seq = torch.argmax(cos_sim_matrix, dim=1).tolist()

        return seq


    def fuzz_matcher(self, sentences1, sentences2):
        seq = []
        for i, a in enumerate(sentences1):
            scores = []
            for j, b in enumerate(sentences2):
                scores.append(fuzz.token_set_ratio(a, b))

            ind = np.argmax(scores)
            seq.append(ind)

        return seq

    def spacy_matcher(self, sentences1, sentences2):
        seq = []
        for i, a in enumerate(sentences1):
            scores = []
            doc1 = self.spacy(a)
            for j, b in enumerate(sentences2):
                doc2 = self.spacy(b)
                scores.append(doc1.similarity(doc2))

            ind = np.argmax(scores)
            seq.append(ind)

        return seq

    def diff_matcher(self, sentences1, sentences2):
        seq = []
        for i, a in enumerate(sentences1):
            scores = []
            for j, b in enumerate(sentences2):
                scores.append(SequenceMatcher(None, a, b).ratio())

            ind = np.argmax(scores)
            seq.append(ind)

        return seq

    def tfidf_matcher(self, sentences1, sentences2):
        seq = []

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences1 + sentences2)

        for i, a in enumerate(sentences1):
            scores = []
            for j, b in enumerate(sentences2):
                scores.append(cosine_similarity(tfidf_matrix[:len(sentences1)][i], tfidf_matrix[len(sentences1):][j]))

            ind = np.argmax(scores)
            seq.append(ind)

        return seq


    def match(self, sentences1, sentences2, minilm=None, bert=None, fuzz=None, spacy=None, diff=None, tfidf=None, pnt=False):
        allseq = []

        if minilm:
            allseq.append(self.miniLM_matcher(sentences1, sentences2))

        if bert:
            allseq.append(self.bert_matcher(sentences1, sentences2))

        if fuzz:
            allseq.append(self.fuzz_matcher(sentences1, sentences2))

        if spacy:
            allseq.append(self.spacy_matcher(sentences1, sentences2))

        if diff:
            allseq.append(self.diff_matcher(sentences1, sentences2))

        if tfidf:
            allseq.append(self.tfidf_matcher(sentences1, sentences2))

        mergedseq = np.array(allseq)

        if pnt:
            print('Individual matches:')
            print(mergedseq)

        #majority voting
        seq = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=mergedseq)
        if pnt:
            print('Consensus match:')
            print(seq)

        return seq


def smooth_sequence(L1):
    N = len(L1)
    M = max(L1)

    # Initialize memo table with infinities
    memo = [[float('inf') for _ in range(M + 1)] for _ in range(N)]

    # Base case
    for j in range(M + 1):
        memo[0][j] = abs(L1[0] - j)

    # DP recurrence
    for i in range(1, N):
        for j in range(M + 1):
            for k in range(j + 1):
                current_loss = abs(L1[i] - j)
                memo[i][j] = min(memo[i][j], memo[i - 1][k] + current_loss)

    # Reconstruct the optimal sequence
    L2 = [0] * N
    last_val = memo[N - 1].index(min(memo[N - 1]))
    L2[N - 1] = last_val
    for i in range(N - 2, -1, -1):
        min_loss = float('inf')
        for j in range(last_val + 1):
            if memo[i][j] + abs(L2[i + 1] - last_val) < min_loss:
                min_loss = memo[i][j] + abs(L2[i + 1] - last_val)
                L2[i] = j
        last_val = L2[i]

    # Compute the total loss
    total_loss = sum([abs(L2[i] - L1[i]) for i in range(N)])

    return L2


def map_text_to_pdfpages(text, pdffile, matcher):
    # extract text from each page
    pdf_reader = PyPDF2.PdfReader(open(pdffile, 'rb'))
    pdf_pages_text = []
    for p in pdf_reader.pages:
        pdf_pages_text.append(p.extract_text())

    # splits = re.split('[.]', text)
    splits = sent_tokenize(text)

    seq = matcher.match(splits, pdf_pages_text, bert=True, minilm=True, fuzz=True, diff=True, tfidf=True)

    # first is always first page
    seq[0] = 0

    smoothed_seq = smooth_sequence(seq)

    return smoothed_seq


def map_page_to_blocks(pagemap, text, gs, files_dir, pdffile, matcher, display):
    splits = sent_tokenize(text)
    blockmap = []
    coords = []

    for pd_ind, pg_num in enumerate(tqdm(np.unique(pagemap))):
        page_pdf = f'{os.path.join(files_dir, str(pg_num))}' + '.pdf'
        page_png = f'{os.path.join(files_dir, str(pg_num))}' + '.png'

        os.system(f'{gs} -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dFirstPage={pg_num + 1} -dLastPage={pg_num + 1} -sOutputFile={page_pdf} {pdffile} {display}')
        os.system(f'{gs} -sDEVICE=png16m -r400 -o {page_png} {page_pdf} {display}')

        doc = fitz.open(page_pdf)
        page = doc[0]

        blocks = page.get_text("blocks")

        good_blocks = []
        good_coords = []
        for b in blocks:
            if len(b[4]) > 0:
                good_blocks.append(b[4])
                good_coords.append(list(b[:4]))

        start = np.where(np.array(pagemap) == pg_num)[0][0]
        end = np.where(np.array(pagemap) == pg_num)[0][-1] + 1
        page_text_splits = splits[start:end]

        seq = matcher.match(page_text_splits, good_blocks, bert=True, minilm=True, fuzz=True, spacy=True, diff=True, tfidf=True, pnt=True)

        smoothed_seq = smooth_sequence(seq)

        blockmap += [[pd_ind, pg_num, s] for s in smoothed_seq]
        coords.append(good_coords)

    return coords, blockmap

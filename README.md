# NLP labs — ARTI 501

Eleven lab notebooks with every task cell filled in and executed. These are the labs from
Natural Language Processing (ARTI 501), cloned from `github.com/MohammedMosuily/NLP_Lab1` …
`NLP_Lab11` and consolidated into this single repository.

| Folder | Notebook | Status |
|---|---|---|
| `NLP_Lab1` | Introduction to NLP | done |
| `NLP_Lab2` | Text pre-processing and regular expressions | done |
| `NLP_Lab3` | N-grams | done |
| `NLP_Lab4` | Classification and evaluation | done |
| `NLP_Lab5` | Text representation | done |
| `NLP_Lab6` | Deep learning for NLP | done |
| `NLP_Lab7` | Large language models | done |
| `NLP_Lab8` | Part of speech tagging | done |
| `NLP_Lab9` | Named entity recognition | done |
| `NLP_Lab10` | Introduction to machine translation | done |
| `NLP_Lab11` | Attention with RNNs | done |

All 168 code cells run with no errors, and the outputs in the notebooks are from real executions,
not transcribed.

## Running them

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn nltk spacy gensim emoji `
    sentencepiece "transformers>=4.50" tensorflow matplotlib ipykernel jupyterlab
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m spacy download en_core_web_sm
.venv\Scripts\python.exe -m ipykernel install --user --name nlp-labs --display-name "NLP Labs (Python 3.13)"
.venv\Scripts\python.exe -m jupyter lab
```

Then select the **NLP Labs (Python 3.13)** kernel.

Python 3.13, not 3.14: **gensim cannot build on 3.14** — the Cython C it ships references
`ma_version_tag`, which 3.14 removed. Not fixable from here.

TensorFlow and PyTorch are both installed, for different labs: Lab 10 and Lab 11 build their Keras
tokenizers with `tensorflow.keras`, Lab 11 trains in torch. TensorFlow 2.21 is the first release
with Python 3.13 wheels, and 2.21 dropped native Windows GPU support, so the Lab 10 model trains on
the CPU. It takes about six minutes.

To re-run everything without a browser:

```powershell
.venv\Scripts\python.exe tools\run_notebooks.py      # all eleven, in order
.venv\Scripts\python.exe tools\check_notebooks.py    # verify, without executing
```

Lab 4 lemmatises 30,000 reviews, Lab 5 trains Word2Vec on 158,000 Simpsons lines, and Lab 10 trains
a translation model for twelve epochs, so those three take a few minutes each. The other eight
finish in under a minute.

### Data

Datasets download themselves on first run via `kagglehub`, into `~/.cache/kagglehub`. Lab 5, Lab 6
and Lab 11 ship their data in-repo. The three Hugging Face models Lab 7 uses are pulled on first run
into `~/.cache/huggingface`.

There is no `kaggle.json` and no API token here, and none is needed: kagglehub fetches
**unrestricted public** datasets anonymously. All six datasets used are in that category. That is a
narrower guarantee than "Kaggle needs no login" — a dataset gated behind a licence acceptance will
fail anonymously, and the fix is `kagglehub.login()`. Anonymous requests are also rate-limited, so a
cold first run that pulls all six can stall where an authenticated one would not.

**Because the data downloads at runtime, the notebooks need internet on first run.** The outputs are
already saved, so they read correctly offline; only re-running requires the network.

## Results

| Lab | And the answer is |
|---|---|
| 1 | IMDB reviews, 50,000 rows, 25,000/25,000 split, mean review 1,309 chars |
| 2 | 1,616 hashtags, 607 unique. Top: `#AAPL` 203, `#aapl` 197 — same tag, counted twice |
| 3 | 202,202 tweets → 163,459 after cleaning. Bigram MLE: `P(is \| pakistan)` = 0.0431, `P(pakistan)` = 0.00616, perplexity of "pakistan" = 162.3 |
| 4 | 3-class Naive Bayes, accuracy 0.8355. `neutral` recall 0.013 — the model never predicts it |
| 5 | doc1 vs doc4 cosine = **1.000** (same bag of words, reordered). `homer` → `abe`, `marge`, `bart` |
| 6 | Task 2 as specified scores **0.500** — chance. Diagnosed below |
| 7 | distilBERT: `NEGATIVE` 0.9997 on "The lecture was difficult to understand." FLAN-T5 summarises |
| 8 | `play` → `VERB`/`VB`; `google` → `NNP` vs `VB`; `table` → `NN` vs `VB` |
| 9 | `Bcd LLC` missed entirely; `Abc & Co.` span swallowed "between". Corrected: 2 `ORG`, 1 `PERSON` |
| 10 | 175,621 pairs → 45,000 training pairs, 12 epochs: BLEU **0.09**, exact match **0.000**. Fluent syntax, wrong content words |
| 11 | The shipped model cannot even fit its 168 questions (accuracy 0.381) — its RNN reads the *batch* axis. Fixed: 1.000 seen, 0.588 unseen |

Lab 3's MLE perplexity is `inf` on purpose: 8,156 of 23,798 held-out bigrams were never seen in
training, and one unseen bigram makes an unsmoothed MLE return infinity. The notebook repeats the
measurement under Laplace smoothing (6,242) so there is a finite number to compare against.

## Lab 10: a translator that gets the grammar and loses the meaning

The task asks for English → French on `devicharith/language-translation-englishfrench`, 175,621
Tatoeba pairs, for an e-commerce use case. The notebook builds it from nothing: normalise the text,
keep short pairs, 6,000-word vocabularies, an encoder LSTM and a decoder LSTM fed the previous
French word, 12 epochs.

Twelve epochs of CPU training buy the *shape* of French — elision, the `ne ... pas` frame,
`est-ce que` inversion, frequent phrases — and not the meaning. On 5,000 held-out pairs the corpus
BLEU is about 0.09 and not one hypothesis matches its reference exactly:

```
it 's rather cold for april .   il fait plutot froid pour un mois d' avril .
                                il est trop jeune pour le faire...
```

The content words are what falls out, for three reasons the notebook lays out in order: the encoder
compresses the whole sentence into one 256-number state (the bottleneck Lab 11 exists to open), the
vocabulary is far too large for 45,000 pairs of training signal, and greedy decoding never
reconsiders a word. Training longer and on five times more data both land in the same BLEU range,
which is the useful negative result: the limit here is capacity and time, not corpus size.

The conclusion for the use case is the honest one. A six-minute from-scratch model is a teaching
device that shows every part of a translation system; shipping product listings needs a pretrained
model (Marian, NLLB, or an API), terminology control for brand names and sizes, and quality
estimation on human-reviewed samples. The notebook keeps what carries over: a held-out split, a
metric, and a look at the sentences the metric came from.

## Lab 11: the attention chart was an artifact

The handout trains a tiny attention RNN on 168 questions with 12 possible answers, then plots the
weights. Taken at face value the plots look right — the top weight lands on a word that matters for
two of the three questions in Task 1. They are not right. `nn.RNN(16, 32)` is built without
`batch_first=True`, so torch reads dim 0 as time; `X` is (questions, words) and the recurrence
therefore walks **across the 168 questions**, treating the 11 word positions as a batch. The model
never reads a sentence. The training loop also never calls `optimizer.zero_grad()`.

The tell is accuracy: **0.381 on its own training data**, for a model with 168 examples and 12
classes. A model allowed to memorise should be at 1.000, and this one cannot be, because nothing in
its recurrence identifies the question.

With the transpose fixed, gradients zeroed, and the padding masked out of the softmax, accuracy goes
to 1.000 (0.588 on 20% held out, against 0.206 for always answering the commonest answer) and the
weights move to content words: `parents?` 1.00, `eat` 0.99. They still are not an explanation.
"so what is your name?" puts 0.81 on `what` and 0.06 on `name?`, the word that decides the answer,
while a paraphrase pair ("who created you?" / "who built you?") attends to the same *slot* in both
(the verb) with a different string each time. The notebook's answer to Task 2 is that attention
routes each question separately; it does not carry meaning across paraphrases, so it cannot be
shipped as an explanation.

## Lab 6 doesn't work, and that's the interesting part

The task says train a network on averaged Word2Vec vectors, with `BCEWithLogitsLoss`, Adam at
`lr=0.001`, 15 epochs, and report the accuracy. Doing exactly that gives **0.500**, with the loss
pinned to ln(2) = 0.693 — the model predicts one class for everything.

It is not the network and not the training loop. Holding the network, loss, optimizer, learning rate
and epoch count completely fixed and swapping only the input features:

```
averaged Word2Vec (200 dims) -> 0.500
TF-IDF           (1000 dims) -> 0.815
```

The embeddings were fit to 8,622 tokens for a 1,786-word vocabulary: **0.02 training tokens per
embedding parameter**. They come out close to random, and mean-pooling about eleven of them gives a
vector whose mean cosine similarity to any *other* review's vector is **0.9998** — every review
points the same way, so there is nothing to classify.

The notebook lists three fixes in order of how much they would help. The headline one: embeddings
need a corpus orders of magnitude larger than 800 short reviews.

## Lab 9's custom model, same story at a smaller scale

Task 3 says train an NER model on one annotated sentence for 20 iterations, save it, and use it.
Doing exactly that gives a model that scores perfectly on that sentence and recognises nothing
anywhere else — handed "Apple Inc. was founded by Steve Jobs in Cupertino, California", it returns
`Inc.` as an organisation and `California` as a product. That is not a bug in the training loop; one
example plus a blank pipeline is all it has.

The notebook shows the one-line change: the same 20 iterations started from `en_core_web_sm` keep
proper names, places and dates. It still degrades — `Aramco` and `Dhahran` come back as `PRODUCT`,
because 20 passes over a single example drag the weights hard toward the new labels. Real NER
training uses hundreds of annotated spans and stops before it forgets.

## Defects found in the lab material

Fixed, because the notebooks do not run outside Colab otherwise. Each one is flagged in place.

**Stopped execution outright**

1. Lab 4 — `pd.read_csv('/content/DisneylandReviews.csv')`. Colab-only path, and the CSV is not in
   the repo. Now fetched with `kagglehub`.
2. Lab 5 — `pd.DataFrame(...)` in the TF-IDF section with no `import pandas as pd` anywhere above it.
3. Lab 5 — `pd.read_csv('/content/True.csv')`. Same Colab path.
4. Lab 6 — `pd.read_csv("07-yelp-dataset.txt")` from the current directory. The file ships in
   `NLP_Lab6/dataset/`.
5. Lab 2 — `nltk.download('punkt')` ran before `import nltk` had ever been executed.
6. Lab 8 — the tagger download was commented out, but `pos_tag()` immediately below needs it.
7. Lab 2 — `#nltk.download('wordnet')` commented out; the lemmatizer below it needs it.
8. Labs 1, 2, 7 — `!pip install X`. The `!` escape runs whatever `pip` is first on `PATH`, i.e. a
   *different* interpreter from the running kernel. Changed to `%pip`, which targets the kernel.
9. Lab 9 — `spacy.load("custom_ner_model")` in the visualization cell, but no such folder ships in
   the repo, and the cell runs *before* Task 3 trains one. The training step it assumes was missing.
10. Lab 11 — `from google.colab import files` (Colab-only, and unused: the path below is hardcoded).
11. Lab 11 — `pd.read_csv("/content/Lab_11_QnA_Dataset_Download.csv")`. Colab-only path; the file
    ships as `NLP_Lab11/dataset/Lab_11_QnA_Dataset.csv`.

**Ran, but silently produced wrong results**

12. Lab 6 — `df["review"] = df["review"].apply(preprocess_text)` was indented *inside* the function,
    after the `return`. It never ran, so every downstream cell worked on raw review text.
13. Lab 6 — the markdown says `stratify=y` keeps the class balance across the split. The call did not
    pass `stratify`.
14. Lab 6 — `torch.FloatTensor(y_train.values)`. Under pandas copy-on-write that array is read-only,
    and torch warns it is undefined behaviour because the optimizer writes back into the same memory.
    Same problem in the TF-IDF tensor conversion.
15. Lab 3 — `lm.perplexity(...)` wants a flat sequence of ngram tuples, not a list of sentences.
    Passing nested lists silently returns `inf`.
16. Lab 11 — `nn.RNN(16, 32)` without `batch_first=True`, fed a (batch, words, features) tensor, so
    the recurrence runs over the batch. Attention then reads a state that never saw a sentence.
17. Lab 11 — the training loop omits `optimizer.zero_grad()`, so the 50 steps accumulate gradients.
18. Lab 11 — attention is a softmax over all 11 padded positions. A short question can hand part of
    its weight to padding tokens, which are not words.
19. Lab 11 — Step 3's markdown says "remove punctuation" and the code does not, so `?` is a
    vocabulary token the model can attend to. Reported, not changed: it is the handout's choice.

**Not bugs, but worth knowing**

- **Lab 7 has no Task 4.** It goes Task 1, Task 2, Task 3, Task 5.
- **Lab 9's Task 3 is numbered 1, 2, 3, 5** — no step 4 either.
- Lab 9's Task 2 asks for entities that were "not identified". The model misses `Bcd LLC` outright
  *and* anchors `Abc & Co.` on "between", dropping the closing quote. Both are fixed.
- Lab 10's Task 1 questions are numbered 1, 3, 5, 8 — there is no 2, 4, 6 or 7.
- Lab 10's model is the handout's `RepeatVector` architecture, which feeds the decoder the same
  English summary at every step, so it never sees the French it has produced. Task 2 uses a decoder
  fed the previous French word instead, and says why.
- Lab 11's 12 answer groups are exactly balanced at 14 questions each, so the majority baseline is
  1/12 on a balanced split.
- Lab 5's Task 5 list is numbered 1, 3, 4.
- Lab 3 asks for "the perplexity of the word (pakistan)". Perplexity is defined for a *sequence*, so
  the honest reading is `1/P(pakistan)` = 162.3. Both that and the bigram value are shown.
- Lab 1's `dataset/README.md` points at a SharePoint folder with no data in it. Lab 1's own task is
  to find a dataset online, so it uses the IMDB CSV.
- Lab 4's neutral class: 3-star reviews are not "in between" in language, so recall on `neutral` is
  near zero. Merging neutral into a binary problem, or class weighting, is the usual fix.

## Provenance

The lab handouts and their structure come from `MohammedMosuily`'s repositories. The task solutions,
the fixes listed above, and the analysis are my own work for ARTI 501.

This repository is public. If these labs are still in rotation, bear in mind that a search for the
lab title will surface it, together with complete solutions.

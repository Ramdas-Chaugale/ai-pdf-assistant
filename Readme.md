# AI PDF Assistant

Upload a PDF, ask questions about it, and get an answer with the passages it came
from shown underneath.

Everything runs on your own machine. No API key, no account, nothing sent to a
third party, so it costs nothing to run and the document never leaves the
computer.

## How it works

Four steps, all local.

1. **Read.** `pypdf` pulls the text out of every page.
2. **Chunk.** The text is cut into fixed slices of 120 characters.
3. **Index.** Each chunk is embedded with `all-MiniLM-L6-v2` and put into a FAISS
   `IndexFlatL2`. The index is built fresh each time a PDF is uploaded.
4. **Answer.** The question is embedded the same way, the four nearest chunks are
   retrieved, and those chunks plus the question go to `flan-t5-large`. The
   prompt tells it to answer only from the context and to say
   "Information not found in document" when the answer is not there.

The retrieved chunks are shown under every answer in expandable panels. That is
the point of the thing: you can check what the model was looking at instead of
taking the answer on trust.

## Running it

Needs Python 3.10 or newer. The models download on first run, about 1 GB, and
are cached after that.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the address Streamlit prints, usually http://localhost:8501.

First run is slow because `flan-t5-large` has to download. After that the models
are cached by Streamlit and stay in memory.

## What it does not do

Worth knowing before you judge the answers.

- **Chunking is naive.** Fixed 120 character slices with no overlap, so sentences
  and words get cut in half and a fact sitting across a boundary can be missed.
  Splitting on sentences with a small overlap is the obvious next step.
- **The context window is small.** The prompt is truncated at 512 tokens, so only
  a few retrieved chunks actually reach the model.
- **`flan-t5-large` is a small model.** Answers are short and sometimes flat. It
  was chosen because it runs on a laptop CPU without a key, not because it is the
  best reader available.
- **Nothing persists.** The index is rebuilt on every upload and lost when the app
  restarts.
- **Scanned PDFs will not work.** There is no OCR, so a PDF of photographed pages
  has no extractable text.

## Built with

Python, Streamlit, FAISS, SentenceTransformers, HuggingFace Transformers,
`flan-t5-large`, pypdf.

## Licence

MIT.

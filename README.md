# Transformer-Lm-From-Scratch

To tryout and run things, checkout the commands in 2nd section.
## Section 1: intro and setup

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

### Setup

#### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

#### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

#### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

---
---
## Section 2: Code execution

- To train the tokenizer the data is avilable under test/fixtures and the compiled output of bpe algorighm is avilable at cs336_basics/artfacts/tokenizer.
- you might be wondering how I got to this output... so under cs336_basics I have scripts folder, and it contains train_save_bpe.py
- this train_save_bpe.py accepts bunch of arguments from commandline... then calles the train_bpe function which is present in tokenizer.train_bpe and then saves it using the functions avilable in tokenizer.io
- to run the tokenizer you can try following command in powershell. (to run command in commandline use \ insted of `)
```bash
uv run python scripts/train_save_bpe.py `
  --input tests/fixtures/corpus.en `
  --vocab_size 500 `
  --output_dir artifacts/tokenizers `
  --name corpus_en_500
```

#### Issue a.1.1
- We have created tokenizer.py file and inside it we have Tokenizer class. to test the tokenizer class we need to run `uv run pytest tests/test_tokenizer.py`. but test_tokenizer.py contains some packages like 'resource' which cannot be loaded on windows, so we need mac or linux to run those.
- other resone to use linux or macOS is `open()` defaults to UTF-8, while on whidows it defaults to `cp1252` encoding. and gpt2_merges.txt contains byte-mapped unicode character by design and cp1252 cannot decoad them.
#### solution a.1.1
- replaceing `with open(merges_path) as f:` with `with open(merges_path, encoding="utf-8") as f:`/

# Study on parameter-efficient fine-tuning techniques for code

## Tasks and datasets.

1. **Defect detection - Devign dataset**
2. **Program translation - XLCoST dataset**
3. **Code generation - Subset of CONCODE dataset**
4. **Code generation - XLCoST dataset**

## Implementation progress.

| **Task**                | **script** | encoder | decoder | encoder-decoder
|-------------------------|------------|---------|---------|-----------------|
| devign_defect_detection | train      | ✅       |  ✅       |      ✅           |
|                         | test       | ✅       |    ✅     |        ✅         |
| xlcost_code_translation | train      | /       | ✅  |     ✅            |
|                         | test       | /       | ✅     |     ✅            |
| xlcost_code_generation  | train      | /       | ✅   | ✅           |
|                         | test       | /       | ✅     | ✅             |
| concode_code_generation | train      | /       | ✅   | ✅           |
|                         | test       | /       | ✅     | ✅             |

## Studied PLMs and LLMs

1. Medium-sized PLMs
    1. CodeBERT
    2. GraphCodeBERT
    3. CodeT5-base
    4. CodeT5+
2. LLMs (from 1B to 12B)
    1. Bloom
    2. InCoder
    3. CodeGen
    4. CodeT5+
    5. PolyCoder
    6. CodeT5-large 

## Setup

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir runs
cd runs
mkdir devign_defect_detection
mkdir xlcost_code_translation
mkdir xlcost_code_generation
mkdir concode_code_generation
```

## Datasets statistics

**Defect detection:**

|       | # Examples |
|-------|:----------:|
| Train |   21,854   |
| Valid |   2,732    |
| Test  |   2,732    |

**Code translation:**

|**Code-Code Pairs**|       | C++  | Java  | Python | C#    | JS    | PHP |
|--------|-------|------|-------|--------|-------|-------|-----|
| Java   | train |9,450 |       |        |       |       |     |
|        | val   |  490 |       |        |       |       |     |
|        | test  |  901 |       |        |       |       |     |
| Python | train |9,139 | 8,991 |        |       |       |     |
|        | val   |  468 | 471   |        |       |       |     |
|        | test  |  878 | 882   |        |       |       |     |
| C#     | train |9,187 | 9,301 | 8,826  |       |       |     |
|        | val   |  488 | 491   | 470    |       |       |     |
|        | test  |  890 | 898   | 877    |       |       |     |
| JS     | train |8,482 | 8,470 | 8,182  | 8,367 |       |     |
|        | val   |  472 | 475   | 459    | 475   |       |     |
|        | test  |  878 | 881   | 864    | 877   |       |     |
| PHP    | train |3,056 | 3,68  | 3,003  | 3,071 | 2,971 |     |
|        | val   |  157 | 158   | 153    | 158   | 157   |     |
|        | test  |  303 | 307   | 304    | 307   | 302   |     |
| C      | train |  402 | 409   | 380    | 394   | 308   | 170 |
|        | val   |   59 | 59    | 59     | 59    | 59    | 55  |
|        | test  |   45 | 49    | 48     | 49    | 49    | 43  |

|       | # Examples |
|-------|:----------:|
| Train |   106K   |
| Valid |   6K    |
| Test  |   11K    |

**Code generation (XLCoST):**

| **NL-Code Pairs** |       | **C++** | **Java** | **Python** | **C#** | **JS** | **PHP** | **C** | **Total** |
|:----------:|:-----:|:-------:|:--------:|:----------:|:------:|:------:|:-------:|:-----:|:---------:|
| **Program**| train |  9,797  |  9,623   |   9,263    | 9,345  | 8,590  |  3,087  |  463  |   50,168   |
|            | valid |   492   |    494   |     472    |   491  |   475  |   158   |   60  |    2,642   |
|            |  test |   909   |    911   |     887    |   899  |   886  |   308   |   51  |    4,851   |

**Code generation (CONCODE):**

|       | # Examples |
|-------|:----------:|
| Train |    100K    |
| Valid |     2K     |
| Test  |     2K     |

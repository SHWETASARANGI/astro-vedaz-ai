# Vedaz AI Engineer Task

## Overview

This project contains solutions for the three required tasks in the Vedaz AI Engineer assessment.

The goal was to build a small pipeline for:

1. Validating and analyzing astrology chat datasets.
2. Generating new training conversations using an LLM.
3. Evaluating AI-generated responses for both quality and safety.

The implementation focuses on safety, automation, and reproducibility.

---

# Repository Structure

```text
vedaz-ai-engineer-task/

├── data/
│   ├── input_chats.jsonl
│   ├── generated_chats.jsonl
│   ├── train.jsonl
│   ├── test.jsonl
│
├── outputs/
│   ├── checker_report.json
│   ├── evaluation_results.csv
│
├── src/
│   ├── checker.py
│   ├── generator.py
│   ├── evaluator.py
│   ├── safety.py
│   ├── llm.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── .env.example
```

---

# Setup

## 1. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Environment Variables

Copy `.env.example` to `.env`.

```bash
cp .env.example .env
```

or on Windows:

```powershell
copy .env.example .env
```

Add your API key:

```env
GEMINI_API_KEY=your_api_key_here
```

---

# Task 1 — Chat Checker

File:

```bash
src/checker.py
```

## Purpose

Validates chat datasets and identifies potentially unsafe conversations.

## Features

* Verifies chat structure:

  * Starts with a system message.
  * Alternates correctly between user and assistant turns.
* Counts approximate chat length.
* Detects duplicate or near-duplicate chats.
* Splits data into:

  * Training set
  * Test set
* Flags safety violations.

## Safety Rules Checked

The checker flags content that:

* Predicts death.
* Predicts serious illness.
* Guarantees financial outcomes.
* Guarantees relationship outcomes.
* Pressures users to purchase remedies.
* Uses fear-based persuasion.
* Claims astrology is certain or infallible.

## Run

```bash
python src/checker.py
```

## Outputs

Generated files:

```text
data/train.jsonl
data/test.jsonl
outputs/checker_report.json
```

---

# Task 2 — Chat Generator

File:

```bash
src/generator.py
```

## Purpose

Generates new astrology conversations using an LLM.

## Features

* Accepts a topic or scenario.
* Requests a complete chat in JSON format.
* Validates generated chats.
* Runs generated chats through the Task 1 safety checker.
* Rejects unsafe outputs automatically.
* Saves only approved conversations.

## Example Topics

* Career delay
* Marriage compatibility
* Relationship concerns
* Family conflict
* Financial uncertainty

## Run

```bash
python src/generator.py
```

## Output

```text
data/generated_chats.jsonl
```

At least 10 valid chats are generated and retained after safety filtering.

---

# Task 3 — Quality Tester

File:

```bash
src/evaluator.py
```

## Purpose

Evaluates AI responses for quality and safety.

## Features

* Sends a set of test questions to an AI model.
* Collects generated answers.
* Uses an automated evaluator to score responses.

## Evaluation Criteria

### Safety

Checks whether the answer:

* Avoids medical predictions.
* Avoids death predictions.
* Avoids guaranteed outcomes.
* Avoids manipulative sales language.

### Helpfulness

Checks whether the answer:

* Is clear.
* Is supportive.
* Addresses the user's concern.

### Honesty

Checks whether the answer:

* Acknowledges the limits of astrology.
* Avoids presenting speculation as fact.

## Run

```bash
python src/evaluator.py
```

## Output

```text
outputs/evaluation_results.csv
```

The CSV contains:

* Question
* Generated Answer
* Safety Score
* Helpfulness Score
* Honesty Score
* Overall Score

---

# Design Choices

## Task 1

I used a hybrid rule-based safety checker.

Keyword and pattern matching were chosen because the prohibited behaviors are relatively specific and can be detected reliably without requiring an additional model call.

Advantages:

* Fast
* Deterministic
* Easy to explain
* Low cost

Limitations:

* May miss subtle unsafe wording.
* May generate false positives when context matters.

With more time, I would combine rule-based detection with an LLM classifier for better contextual understanding.

---

## Task 2

Generated chats are validated immediately after generation.

Every generated conversation is passed through the Task 1 checker before being saved.

This creates a simple generation → validation → storage pipeline and prevents unsafe examples from entering the dataset.

With more time, I would implement automatic regeneration when a chat fails validation.

---

## Task 3

I used automated LLM-based grading because qualities such as warmth, honesty, and helpfulness are difficult to measure using rules alone.

The evaluator scores each response using predefined criteria and produces a structured results table.

With more time, I would:

* Add human-reviewed benchmark examples.
* Improve scoring consistency.
* Compare multiple models side-by-side.

---

# Submission Contents

This submission includes:

* Task 1: Chat Checker (`checker.py`)
* Task 2: Chat Generator (`generator.py`)
* Task 3: Quality Tester (`evaluator.py`)
* Supporting modules:

  * `safety.py`
  * `llm.py`
  * `utils.py`
* Generated dataset:

  * `generated_chats.jsonl`
* Dataset splits:

  * `train.jsonl`
  * `test.jsonl`
* Checker report:

  * `checker_report.json`
* Evaluation results:

  * `evaluation_results.csv`

---

# Notes

The project was developed using the Gemini API through the OpenAI-compatible endpoint. During testing, Gemini free-tier quota limits occasionally returned HTTP 429 responses. The scripts include retry handling, but a paid API tier or refreshed quota may be required for large evaluation runs.

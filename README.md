# Twitter Sentiment Analysis using BERT

This project performs sentiment analysis on Twitter data using a fine-tuned BERT model from Hugging Face Transformers. The model classifies tweets into three categories:

- Negative
- Neutral
- Positive

---

## Dataset

- Total Tweets: 162,980
- Features:
  - `clean_text`
  - `category`

### Label Mapping

| Original | Sentiment | Training Label |
|----------|------------|----------------|
| -1 | Negative | 0 |
| 0 | Neutral | 1 |
| 1 | Positive | 2 |

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Model Architecture

The project uses the pretrained **BERT Base Uncased** model from Hugging Face Transformers for sentiment classification.

### BERT Configuration

- 12 Transformer Encoder Layers
- 12 Self-Attention Heads
- 768 Hidden Dimensions
- ~110 Million Parameters
- Maximum Sequence Length: 128

### Architecture Flow

```python
Input Text
   ↓
BERT Tokenizer
   ↓
BERT Encoder (12 Layers, 12 Attention Heads)
   ↓
[CLS] Token Representation (768-dimensional vector)
   ↓
Dropout Layer
   ↓
Fully Connected Linear Layer
   ↓
Sentiment Prediction
```

The final `[CLS]` token embedding produced by BERT is used as the contextual representation of the tweet and passed through a dropout layer and classifier for sentiment prediction.

---

## Training Details

- Epochs: 3
- Batch Size: 32
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Max Sequence Length: 128

---

## Results

### Final Test Accuracy

```python
98.17%
```

### Classification Performance

| Sentiment | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Negative | 0.97 | 0.97 | 0.97 |
| Neutral | 0.99 | 0.98 | 0.99 |
| Positive | 0.98 | 0.99 | 0.98 |


---

## Future Improvements

- Deploy using Streamlit
- Hyperparameter tuning
- Try RoBERTa or DistilBERT
- Add multilingual sentiment analysis

---

## Conclusion

The fine-tuned BERT model achieved high accuracy on Twitter sentiment classification and demonstrated strong performance across all sentiment classes using transformer-based NLP techniques.

---

## Author

Uma Nishikant Patil

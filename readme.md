<h1> Introduction </h1>

In this project we want to implement a simple alpha signal generation programme that takes as input new headlines (regarding bitcoin, but the particular choice is just for flavour) employing sentiment analysis. The first step in such an implementation would be to scour websites to retrieve headlines. As this a standard task as soon as you have API access to your website of choice, we eschew it in this presentation. Instead we will take the [dataset](https://www.kaggle.com/datasets/notlucasp/financial-news-headlines/) as given, that comprises of financial news headlines from the CNBC, the Guardian, and Reuters.

<h3>Filtering and Cleanup of Relevant Articles</h3>

To implement this filter we will use ``` pandas ``` and ```NLTK``` libraries. The following code does the job

```
def filter_headlines(csv_file, topic):
    """Filters out headlines about a specific topic from a CSV file.

    Args:
        csv_file: The path to the CSV file containing the headlines.
        topic: The topic to filter out.

    Returns:
        A dataframe with the filtered news.
    """

    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(csv_file)
    df = df.dropna()
    # Filter out the headlines that contain the topic.
    filtered_headlines = pd.DataFrame(df.loc[df['Headlines'].str.contains(topic, case=False), 'Headlines'].reset_index(drop=True))
    return filtered_headlines

# Using it for bitcoin:
topic = 'bitcoin'
guardian_file = 'guardian_headlines.csv'
cnbc_file = 'cnbc_headlines.csv'
reuters_file = 'reuters_headlines.csv'

df_g = filter_headlines(guardian_file, topic)
df_c = filter_headlines(cnbc_file, topic)
df_r = filter_headlines(reuters_file, topic)

headlines = pd.concat([df_g,df_c,df_r], ignore_index=True)

# Use a lambda function to remove stopwords
headlines['Headlines'] = headlines['Headlines'].apply(lambda headline: ' '.join([word for word in headline.split() if word not in nltk.corpus.stopwords.words('english')]))

# Convert all text to lowercase
headlines['Headlines'] = headlines['Headlines'].apply(str.lower)

```
<h2>Sentiment Analysis via NLTK </h2>

To set a baseline for our algorithm later on, we first see how ```NLTK``` performs in at this task. The following code can be used to implement this. This is easily done.

```
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Generate alpha signals from the news headlines data
sentiment_scores = {}
for headline in headlines['Headlines']:
    for word in headline.split():
        sentiment_score = sia.polarity_scores(word)
        sentiment_scores[word] = sentiment_score['compound']

headlines['sentiment_score'] = headlines['Headlines'].apply(lambda headline: sum([sentiment_scores[word] for word in headline.split()]))
alpha_signal = headlines['sentiment_score'].sum()
print(alpha_signa)
```
Here we ascribe to the naive approach of summing the sentiment score for all the headlines to get a statistic. The sign here will tell us whether we sell or buy bitcoin, with the magnitude giving some idea of how big a transaction it ought to be. For our chosen dataset we get the figure $-4.5122$. We now repeat the experiment with our own CNN.

<h2>CNN for Sentiment Analysis</h2>

To implement our own CNN, we will again rely on ``` NLTK ```. Using the previous algorithm we will ascribe sentiment scores to all the headlines that were filtered out when we were looking for bitcoin. This data will serve as the training and validation set for our model (we use the conventional 80/20 split). This choice is due to our insistence that our model is trained on financial headlines. The following code gets the data prepped:
```
## Read in the scored data
hd_pt = pd.read_csv('Scored_Headlines.csv')

#a# Split into training and validation sets
x_train = hd_pt['Headlines'][:int(len(hd_pt) * 0.8)]
x_val = hd_pt['Headlines'][int(len(hd_pt) * 0.8):]
y_train = hd_pt['sentiment_score'][:int(len(hd_pt) * 0.8)]
y_val = hd_pt['sentiment_score'][int(len(hd_pt) * 0.8):]
## Initialise BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

## Converting the data into lists of strings.
x_train = x_train.to_list()
x_val = x_val.to_list()

##Tokenising the data
x_train_enc = tokenizer(x_train, truncation=True, padding=True)
x_val_enc = tokenizer(x_val, truncation=True, padding=True)

## Converting the tokenised data into an array
x_train_enc = np.array(x_train_enc['input_ids'])
x_val_enc = np.array(x_val_enc['input_ids'])
```
We are now in a position to set up our model:

```
## Create a BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

## Add a convolutional layer, max pooling layer, flatten the output, add a dense layer, and an output layer
model.classifier = tf.keras.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
  tf.keras.layers.MaxPooling1D(pool_size=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')])

## Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
Now we train our model

```
## Create a BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

## Add a convolutional layer, max pooling layer, flatten the output, add a dense layer, and an output layer
model.classifier = tf.keras.Sequential([
    tf.keras.layers.Reshape((model.config.hidden_size, 1)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

## Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

## Train the data
model.fit(x_train_enc, y_train, epochs=10, validation_data=(x_val_enc, y_val))

## Evaluate the model
score = model.evaluate(x_val_enc, y_val)

## Check the accuracy
print('Accuracy:', score[1])
```
We are now able to compute the scores:

```
## Tokenize the data we need to check
hd['HD']=hd['Headlines'].apply(lamda hd: tokenizer([hd], truncation=True, padding=True))

## Convert tokens to np array
hd['HD2']=hd['HD'].apply(lambda hd: np.array(hd['input_ids']))

## Prediction
hd['predict']=hd['HD2'].apply(model.predict)

#Import softmax
from scipy.special import softmax

## Apply to prediction
hd['score']=hd['predict'].apply(lambda pred: softmax(pred.logits))
```

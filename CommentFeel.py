import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import pandas as pd
import googleapiclient.discovery
import plotly.express as px 


# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


# Set up the YouTube API service
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyC4Vx8G6nm3Ow9xq7NluTuCCJ1d_5w4YPE"  # Replace with your actual API key

youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Function to fetch comments for a video ID
def scrape_comments(video_id):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    response = request.execute()


    comments = []

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['textDisplay']
        ])

    comments_df = pd.DataFrame(comments, columns=['comment'])

    # df.head(10).

    return comments_df


# Function to extract video ID from YouTube URL
def extract_video_id(video_url):
    match = re.search(r'(?<=v=)[\w-]+', video_url)
    if match:
        return match.group(0)
    else:
        st.error("Invalid YouTube video URL")

# Function to fetch YouTube comments for a video ID
def fetch_comments(video_id):
    # Example using youtube-comment-scraper-python library
    comments = scrape_comments(video_id)
    return comments

# Function to analyze sentiment for a single comment
def analyze_sentiment(comment):
    tokens = tokenizer.encode(comment, return_tensors="pt", max_length=512, truncation=True)
    # input_ids = tokens['input_ids']
    # attention_mask = tokens['attention_mask']

    # result = model(input_ids, attention_mask=attention_mask)
    result = model(tokens)

    sentiment_id = torch.argmax(result.logits) + 1
    if(sentiment_id > 3):
        sentiment_label = "Positive"
    elif(sentiment_id < 3):
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return sentiment_label


def main():
    st.title("YouTube Comments Sentiment Analysis")
    st.write("Enter a YouTube video link below:")

    video_url = st.text_input("YouTube Video URL:")
    if st.button("Extract Comments and Analyze"):
        video_id = extract_video_id(video_url)
        if video_id:
            comments_df = fetch_comments(video_id)
            # Comments is a dataframe of just the comments text
            # st.write("Top 100 Comments extracted\n", comments_df)
            comments_df['sentiment'] = comments_df['comment'].apply(lambda x: analyze_sentiment(x[:512]))
            sentiment_counts = comments_df['sentiment'].value_counts()
            positive_count = comments_df['sentiment'].value_counts().get('Positive', 0)
            negative_count = comments_df['sentiment'].value_counts().get('Negative', 0)
            neutral_count = comments_df['sentiment'].value_counts().get('Neutral', 0)

                # Create pie chart in col2 with custom colors
            fig_pie = px.pie(values=[positive_count, negative_count, neutral_count],
                            names=['Positive', 'Negative', 'Neutral'],
                            title='Pie chart representations',
                            color=sentiment_counts.index,  # Use sentiment categories as colors
                            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'})
            st.plotly_chart(fig_pie, use_container_width=True)

            # Create bar chart below the pie chart with custom colors
            fig_bar = px.bar(x=sentiment_counts.index, y=sentiment_counts.values,
                            labels={'x': 'Sentiment', 'y': 'Count'},
                            title='Bar plot representations',
                            color=sentiment_counts.index,  # Use sentiment categories as colors
                            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'})
            st.plotly_chart(fig_bar)


if __name__ == "__main__":
    main()


from gc import collect
from turtle import width
import streamlit as st
import os
import re
import urllib.parse as p
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import requests
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import csv
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from torch import true_divide
from wordcloud import WordCloud
from collections import Counter
from streamlit.components.v1 import components
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import words as nltk_words
from gensim import corpora, models
import gensim.downloader as api
import API_KEY_FOLDER.api_keys
# Run the below downloads only once
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('words')
# nltk.download('vader_lexicon')


api_key = API_KEY_FOLDER.api_keys.YOUTUBE_API_KEY
newsapi_key = API_KEY_FOLDER.api_keys.NEWSAPI_KEY
youtube = build('youtube', 'v3', developerKey=api_key)
os.environ["OPENAI_API_KEY"] = API_KEY_FOLDER.api_keys.OPENAI_API_KEY

# Initialize OpenAI Chat model
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

# Function to extract video ID from YouTube URL provided
def extract_video_id(url):
    # Regular expression to match YouTube video ID in different URL formats
    pattern = r'(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|.*&v=))([\w-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

# Function will take the video ID and extract all the comments as well as their likes from the URL provided and saves it to the video_comments.csv file
def retrieve_video_details(video_id):
    """
    Retrieve video details such as description, comments, and likes for a YouTube video and save them to a CSV file.

    Args:
    - video_id (str): The ID of the YouTube video.

    Returns:
    - pd.DataFrame: DataFrame containing the retrieved data (Video ID, Description, Comment, Likes, Replies).
    - int: Total number of views.
    - int: Total number of likes.
    """
    if video_id:
        # Retrieve video details
        video_response = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()

        if 'items' in video_response and video_response['items']:
            video_info = video_response['items'][0]

            # Get video details
            video_title = video_info['snippet']['title']
            video_description = video_info['snippet']['description']
            total_views = int(video_info['statistics'].get('viewCount', 0))
            total_likes = int(video_info['statistics'].get('likeCount', 0))

            # Retrieve comments for the video
            comments_list = []
            nextPageToken = None

            while True:
                comments = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    textFormat='plainText',
                    maxResults=1000,
                    pageToken=nextPageToken
                ).execute()

                for comment in comments['items']:
                    comment_text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                    comment_likes = comment['snippet']['topLevelComment']['snippet']['likeCount']
                    comment_replies = comment['snippet']['totalReplyCount']
                    comments_list.append({'Comment': comment_text, 'Likes': comment_likes, 'Replies': comment_replies})

                nextPageToken = comments.get('nextPageToken')
                if not nextPageToken:
                    break

            # Create DataFrame from the collected data
            comments_df = pd.DataFrame(comments_list)

            # Save DataFrame to CSV file
            comments_df.to_csv('Data/video_comments.csv', index=False)

            return comments_df, total_views, total_likes, video_title
        else:
            return None, None, None, None
    else:
        print("Invalid YouTube video ID.")
        return None, None, None, None
# Function 'get_wordnet_pos' and 'preprocess_text'  new category of preprocessed comments column which will remove lowercasing, removing special characters,
# numbers, links, filtering non-English words, removing stop words, performing part-of-speech tagging, lemmatizing tokens, and joining them back into a string
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'  # Adjective
    elif tag.startswith('V'):
        return 'v'  # Verb
    elif tag.startswith('N'):
        return 'n'  # Noun
    elif tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun if POS tag not found
def preprocess_text(comment):
    # Lowercase the text
    if isinstance(comment, str):
        # Lowercase the text
        comment = comment.lower()

        # Remove special characters and links
        comment = re.sub(r'http\S+|www\S+|https\S+', '', comment, flags=re.MULTILINE)
        comment = re.sub(r'[^\w\s]', '', comment)

        # Check if the comment contains only numbers or if its length is less than 10 characters
        if comment.isdigit() or len(comment) < 10:
            return ''  # Return empty string
        else:
            return comment.strip()  # Strip leading and trailing spaces
    else:
        return ''  # Return empty string for missing values

# Function to cluster them into categories based on content.
def preprocess_and_cluster_comments(comments_df, num_clusters=5):
    """
    Preprocess comments and cluster them into categories based on content.

    Args:
    - comments_df (pd.DataFrame): DataFrame containing comments data.
    - num_clusters (int): Number of clusters/topics.

    Returns:
    - pd.DataFrame: DataFrame containing comments with cluster category.
    """
    # Step 1: Preprocessing
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(comments_df['processed_comment'])  # Extract comments from the DataFrame

    # Step 2: Choose the Number of Clusters (Topics)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Step 3: Cluster the Data
    clusters = kmeans.fit_predict(X)

    # Manually assign descriptive labels to each cluster based on the content of the comments
    cluster_labels = {
        0: "Political Elections",
        1: "Government Policies",
        2: "Social Issues",
        3: "Economic Concerns",
        4: "International Relations"
    }

    # Add a new column 'Cluster Category' to store the cluster category names
    comments_df['Cluster Category'] = ''

    # Assign cluster category names to each comment
    for cluster_label, label in cluster_labels.items():
        cluster_indices = [i for i, cluster in enumerate(clusters) if cluster == cluster_label]
        comments_df.loc[cluster_indices, 'Cluster Category'] = label

    # Save the DataFrame back to the CSV file
    comments_df.to_csv('video_comments.csv', index=False)

    return comments_df

# Function to do sentiment analysis on the comments present in the dataframe
def analyze_sentiment(comments_df, output_file):
    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    # Function to get sentiment score using VADER
    def get_vader_sentiment(text):
        sentiment = sia.polarity_scores(text)
        return sentiment['compound']

    # Apply sentiment score calculation to each comment
    comments_df['sentiment_score'] = comments_df['processed_comment'].apply(get_vader_sentiment)

    # Function to categorize sentiment
    def categorize_sentiment(score):
        if score > 0:
            return 'Positive'
        elif score < 0:
            return 'Negative'
        else:
            return 'Neutral'

    # Apply sentiment categorization to each comment
    comments_df['sentiment_category'] = comments_df['sentiment_score'].apply(categorize_sentiment)

    # Create a bar chart
    sentiment_counts = comments_df['sentiment_category'].value_counts()
    plt.bar(sentiment_counts.index, sentiment_counts.values)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis of Comments')
    plt.show()
    # Save the bar chart to a PNG file
    plt.savefig('Output/Sentiment_Analysis_Bar_Chart.png')
    # Close the plot to release memory
    plt.close()

    # Save the DataFrame to a CSV file
    comments_df.to_csv("Data/video_comments.csv", index=False)

# Function to extract n-grams from text
def extract_phrases(text, n):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return list(ngrams(filtered_tokens, n))

# Function to get top phrases based on sentiment
def top_phrases(sentiment_comments, n, top_n):
    phrases = []
    for comment in sentiment_comments:
        comment_phrases = extract_phrases(comment, n)
        phrases.extend(comment_phrases)
    phrase_freq = Counter(phrases)
    return phrase_freq.most_common(top_n)    

# Function to generate positive and negative word clouds
def generate_wordcloud_from_phrases(phrase_freq, title, output_file):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(phrase_freq)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    output_path = os.path.join('C:/SFU/Big Data Lab 2/Project/Output/', output_file)  
    plt.savefig(output_path)
    plt.show()
    plt.close()
def convert_to_dict(phrase_list):
    freq_dict = {' '.join(phrase): freq for phrase, freq in phrase_list}
    return freq_dict

# Function to calculate the engagement score of the video
def calculate_engagement_score(total_likes, total_views,comments_df):
    """
    Calculate engagement score for comments based on the ratio of likes to views.

    Args:
    - comments_df (pd.DataFrame): DataFrame containing comments data.
    - total_likes (int): Total number of likes for the video.
    - total_views (int): Total number of views for the video.

    Returns:
    - pd.DataFrame: DataFrame containing comments with engagement score.
    """
    # Calculate ratio of likes to views
    total_comments = len(comments_df['processed_comment'])  # Get the total number of comments indirectly from the DataFrame
    engagement_score = min(((total_likes + total_comments) / total_views) * 100, 100)
    return engagement_score

# Function to calculate the sentiment score of each comment and give top 3 engaging comments
def analyze_comments_sentiment_and_engagement(comments_df):
    """
    Analyze the sentiment of comments in a DataFrame and determine the overall response.

    Args:
    - comments_df (pd.DataFrame): DataFrame containing comments and their sentiment scores.

    Returns:
    - str: Overall response based on the sentiment analysis.
    - int: Number of positive comments.
    - int: Number of negative comments.
    - int: Number of neutral comments.
    """
    # Count the number of positive, negative, and neutral comments within the top 25 engaging comments
    comments_df['engagement_score'] = comments_df['Likes'] + comments_df['Replies'] * 0.5
    top_engaging_comments = comments_df.nlargest(25, 'engagement_score')
    positive_count = sum(top_engaging_comments['sentiment_score'] > 0.2)
    negative_count = sum(top_engaging_comments['sentiment_score'] < 0)
    neutral_count = sum(top_engaging_comments['sentiment_score'].between(-0.2, 0.2))  # Sentiment score between -0.2 and 0.2 is considered neutral
    comments_df.to_csv('video_comments.csv', index=False)

    # Determine the overall response based on counts
    if positive_count > negative_count and positive_count > neutral_count:
        overall_response = 'Positive'
    elif negative_count > positive_count and negative_count > neutral_count:
        overall_response = 'Negative'
    else:
        overall_response = 'Neutral'
    # Plot bar chart
    labels = ['Positive', 'Negative', 'Neutral']
    counts = [positive_count, negative_count, neutral_count]
    plt.bar(labels, counts, color=['green', 'red', 'blue'])
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Comments')
    plt.title('Sentiment Analysis of Top 25 Engaging Comments')
    plt.savefig('Output/Sentiment_Analysis_Bar_Chart.png')  # Save the bar chart as a PNG file
    plt.close()

    return overall_response, positive_count, negative_count, neutral_count

# Function to return the top 3 engaging comments
def top_3_engaging_comments(df):
    # Sort DataFrame by 'engagement_score' column in descending order and get the top 3 rows
    top_comments_df = df.sort_values(by='engagement_score', ascending=False).head(3)
    return top_comments_df

# Function to Generate top 3 topics discussed in the comments section
def extract_top_topics(comments_df):
    # Create a prompt for all comments
    prompt = "Summarize the content of all comments and provide the top 3 topics discussed in the comment section. \
    Each topic will have atmost 7 words. \
    The response should start with 'List of topics discussed: and will have number 1. <topic1>\n2. <topic2>\n3. <topic3>'\n\n"
    prompt += "\n\n".join([f"Comment: {comment}" for comment in comments_df['processed_comment']])

    # Generate response from AI model
    response = chat([HumanMessage(content=prompt)])
    
    sections = response.content.strip().split("\n")

    # Extract the headings of the top 3 topics
    headings = []
    for section in sections:
        # Find the index of the first digit or hyphen
        index = next((i for i, c in enumerate(section) if c.isdigit() or c == '-'), -1)
        if index != -1:
            # Add the substring after the digit or hyphen and space
            headings.append(section[index + 2:].strip())
        else:
            # If neither digit nor hyphen found, add the entire section
            headings.append(section.strip())

    return headings[1:]  # Skip the first line which contains "Top 3 Topics Discussed:"

# Function to get top 3 news related to the 3 discussed topics
def fetch_articles(topics, newsapi_key):
    """
    Fetch articles related to the given topics using NewsAPI.

    Args:
    - topics (list): List of topics.
    - newsapi_key (str): API key for accessing NewsAPI.

    Returns:
    - pd.DataFrame: DataFrame containing fetched articles with columns ['Topic', 'Title', 'URL'].
    """
    # Base URL for fetching news from NewsAPI
    base_url = "https://newsapi.org/v2/everything"

    all_articles = []
    for topic in topics:
        params = {
            'q': topic,
            'sortBy': 'relevancy',
            'pageSize': 1,  # Fetch only one article per topic
            'apiKey': newsapi_key,
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            articles = response.json().get('articles', [])

            if articles:
                article = articles[0]  # Get the first article
                all_articles.append({
                    'Topic': topic,
                    'Title': article['title'],
                    'URL': article['url']
                })
            else:
                print(f"No articles found for topic '{topic}'")
        else:
            print(f"Failed to fetch articles for topic '{topic}'. Status code: {response.status_code}")

    df_articles = pd.DataFrame(all_articles, columns=['Topic', 'Title', 'URL'])
    return df_articles

# Function to resize the images
def resize_image(image_path, width, height):
    image = Image.open(image_path)
    resized_image = image.resize((width, height))
    return resized_image

def format_number(num):
    if num >= 1000000:  # If number is >= 1 million
        return f"{num / 1000000:.1f}M"  # Convert to millions with one decimal place
    elif num >= 1000:  # If number is >= 1 thousand
        return f"{num / 1000:.1f}K"  # Convert to thousands with one decimal place
    else:
        return str(num)  # Return the original number if it's less than 1000

def main():
    st.title("📑Video Analytics Dashboard📑")
    # Video URL input field in the first column
    flag = False
    # Horizontal layout for input field and button
    col1, col2 = st.columns([2, 1])
    # Input field for video URL in the first column
    with col1:
        video_url = st.text_input("", key="video_url", placeholder="Enter Video URL here...")
    # "Go" button in the second column
    with col2:
        st.write("")
        st.write("")
        if st.button("Go"):
            if video_url:
                flag = True   
            # Retrieve video ID from the URL
    if flag:
        video_id = extract_video_id(video_url)
        # Call function to analyze video analytics

        # Retrieve video details
        comments_df, total_views, total_likes, video_title = retrieve_video_details(video_id)
        # Define the HTML code for the card
        card_html = """
        <div class="tilecard">
        <div class="card-body">
            <h3 class="card-title">Video Title</h3>
            <p class="card-text">{}</p>
        </div>
        </div>
        """
        # Render the card using st.markdown
        st.markdown(card_html.format(video_title), unsafe_allow_html=True)
        if comments_df is not None:
            # Display KPIs
            #Row 1
            formatted_likes = format_number(total_likes)
            formatted_views = format_number(total_views)
            formatted_comments = format_number(len(comments_df))
            st.markdown("### Key Performance Indicators (KPIs)📈")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Likes", formatted_likes)
            col2.metric("Views", formatted_views)
            col3.metric("Comments", formatted_comments)

            # Preprocess the comments
            comments_df['processed_comment'] = comments_df['Comment'].apply(preprocess_text)

            # create a cluster category
            comments_df = preprocess_and_cluster_comments(comments_df)
            
            # analyze the sentiment of the video
            analyze_sentiment(comments_df, 'video_comments.csv')

            # Filter positive and negative comments
            positive_comments = comments_df[comments_df['sentiment_category'] == 'Positive']['processed_comment']
            negative_comments = comments_df[comments_df['sentiment_category'] == 'Negative']['processed_comment']
            # Get top phrases for positive comments (2 and 3 words)
            top_positive_phrases_2_words = top_phrases(positive_comments, 2, 100)
            # Get top phrases for negative comments (2 and 3 words)
            top_negative_phrases_2_words = top_phrases(negative_comments, 2, 100)

            positive_phrases_2_dict = convert_to_dict(top_positive_phrases_2_words)
            negative_phrases_2_dict = convert_to_dict(top_negative_phrases_2_words)

            # Define output file paths
            output_file_positive = 'wordcloud_positive.png'
            output_file_negative = 'wordcloud_negative.png'

            # Generate word cloud for positive phrases and save as PNG
            generate_wordcloud_from_phrases(positive_phrases_2_dict, 'Word Cloud for Positive Phrases', output_file_positive)
            generate_wordcloud_from_phrases(negative_phrases_2_dict, 'Word Cloud for Negative Phrases', output_file_negative)       

            # Calculate and display engagement rate
            engagement_score = calculate_engagement_score(total_likes, total_views, comments_df)
            engagement_score = "{:.2f}%".format(engagement_score)
            col4.metric("Engagement Rate", engagement_score)
            # Analyze sentiment score of each comment and give number of positive, negative and neutral comments 
            overall_response, positive_count, negative_count, neutral_count = analyze_comments_sentiment_and_engagement(comments_df)
            col5.metric("Public Emotion", overall_response)


            # Extract top 3 topics discussed
            comments_df=comments_df.nlargest(100, 'engagement_score')
            headings = extract_top_topics(comments_df)
            df_articles = fetch_articles(headings, newsapi_key)
           

            # Create bullet points for topics
            topics_bullet_points = "<br>".join([f" &#8226 {topic}" for topic in headings[:3]])

            # Extract top 3 articles titles and URLs
            top_3_article_titles = df_articles['Title'].head(3)
            top_3_article_urls = df_articles['URL'].head(3)
            
            #checkpoint
            comments_df.to_csv("Data/video_comments.csv", index=False)
            articles_bullet_points = "<br>".join([f" &#8226; <a style='color: blue; text-decoration: underline;' href='{url}' target='_blank'>{title}</a>" for title, url in zip(top_3_article_titles, top_3_article_urls)])
            top_comments_df = top_3_engaging_comments(comments_df)
            top_comments_bullet_points = "<br>".join([f" &#8226 {processed_comment}" for processed_comment in top_comments_df['processed_comment']])
           
            html_code = f"""
            <style>
                .row {{
                    display: flex;
                    flex-direction: row;
                    justify-content: space-between;
                }}
                .column {{
                    flex: 1;
                    margin: 5px;
                }}
                .card {{
                    background-color: rgba(0, 0, 0, 0.25);
                    color: #ffffff;
                    padding: 10px;
                    width: auto;
                    height: 300px;
                    border-radius: 5px;
                    overflow: auto; /* Enable scroll if content exceeds height */
                }}
                .card a {{
                    color: #ffffff; /* Set link color */
                    text-decoration: none; /* Remove underline */
                }}
            </style>
            <div class="row">
                <div class="column">
                    <div class="card">
                            <h4>Top 3 topics discussed</h4>
                            <p>{topics_bullet_points}</p>
                            <!-- Insert a blank line -->
                            <br>
                    </div>
                </div>
                <div class="column">
                    <div class="card">
                        <h4>Latest News Articles</h4>
                        <p>{articles_bullet_points}</p>
                    </div>
                </div>
                <div class="column">
                    <div class="card">
                        <h4>Top Engaging Comments</h4>
                        <p>{top_comments_bullet_points}</p>
                    </div>
                </div>
            </div>
            """
            st.markdown(html_code, unsafe_allow_html=True)
            
           # Define image paths
            sentiment_analysis_image_path = "Output/Sentiment_Analysis_Bar_Chart.png"
            positive_word_cloud_image_path = "Output/wordcloud_positive.png"
            negative_word_cloud_image_path = "Output/wordcloud_negative.png"

            # Resize images
            resized_sentiment_analysis_image = resize_image(sentiment_analysis_image_path, 600, 400)
            resized_positive_word_cloud_image = resize_image(positive_word_cloud_image_path, 600, 400)
            resized_negative_word_cloud_image = resize_image(negative_word_cloud_image_path, 600, 400)

            # Display resized images
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("<h4>Sentiment Analysis</h4>", unsafe_allow_html=True)
                st.image(resized_sentiment_analysis_image)

            with col2:
                st.markdown("<h4>Positive Word Cloud</h4>", unsafe_allow_html=True)
                st.image(resized_positive_word_cloud_image)

            with col3:
                st.markdown("<h4>Negative Word Cloud</h4>", unsafe_allow_html=True)
                st.image(resized_negative_word_cloud_image)
                        
            

if __name__ == "__main__":
    main()

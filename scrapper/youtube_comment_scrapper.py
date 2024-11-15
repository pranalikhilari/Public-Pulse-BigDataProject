import pandas as pd

urls = pd.read_excel('/data_collection/comments_data/url_data.xlsx')

# Replace 'YOUR_API_KEY' with your actual YouTube Data API key
api_key = ''


video_details_file_path = 'data_collection/comments_data/video_details_1.csv'
comments_file_path = 'data_collection/comments_data/video_comments_1.csv'

import json
import pandas as pd
from googleapiclient.discovery import build
import re

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    # Regular expression to match YouTube video ID in different URL formats
    pattern = r'(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|.*&v=))([\w-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None
    
for video_url in urls['URL']:
    print(video_url)
    video_id = extract_video_id(video_url)

    youtube = build('youtube', 'v3', developerKey=api_key)

    # Call the API to get video details
    response = youtube.videos().list(
        part='snippet,statistics',
        id=video_id
    ).execute()

    # Extract relevant information
    video_info = response['items'][0]
    title = video_info['snippet']['title']
    description = video_info['snippet']['description']
    views = video_info['statistics']['viewCount']
    likes = video_info['statistics']['likeCount']

    # Create a DataFrame
    video_data = pd.DataFrame({
        'Video ID': [video_id],
        'Title': [title],
        'Description': [description],
        'Views': [views],
        'Likes': [likes]
    })

    # Read existing CSV file, if it exists

    try:
        existing_data = pd.read_csv(video_details_file_path)
        combined_data = pd.concat([existing_data, video_data], ignore_index=True)
    except FileNotFoundError:
        combined_data = video_data

    combined_data.to_csv(video_details_file_path, index=False)


    # code to extract the comments
    import os

    # Check if the CSV file already exists
    if os.path.isfile(comments_file_path):        
        existing_data = pd.read_csv(comments_file_path)
    else:        
        existing_data = pd.DataFrame(columns=['Video ID', 'Comment', 'Likes', 'Replies'])

    # Assuming 'video_id' and 'comments_df' are defined earlier in your code
    if video_id:        
        comments_list = []
        
        nextPageToken = None
        total_comments = 0

        while True:            
            comments = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=1000,  
                pageToken=nextPageToken
            ).execute()

            # Process and store comments in the list
            for comment in comments['items']:
                comment_text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
                likes = comment['snippet']['topLevelComment']['snippet']['likeCount']
                replies = comment['snippet']['totalReplyCount']  # Total number of replies
                comments_list.append({'Comment': comment_text, 'Likes': likes, 'Replies': replies})
                total_comments += 1

            # Check if there are more pages of comments
            nextPageToken = comments.get('nextPageToken')
            if not nextPageToken:
                break  

            print(f"Collected {total_comments} comments so far...")

        print(f"Total comments collected: {total_comments}")

        # Create a DataFrame from the list of comments
        new_comments_df = pd.DataFrame(comments_list)

        # Concatenate the new comments with existing data
        combined_data_comments = pd.concat([existing_data, new_comments_df], ignore_index=True)
        combined_data_comments['Video ID'] = video_id

        # Write the combined DataFrame to the CSV file
        combined_data_comments.to_csv(comments_file_path, index=False, encoding='utf-8')

        print("All comments saved")
    else:
        print("Invalid YouTube video URL or video ID not found.")

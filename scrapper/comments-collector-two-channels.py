from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd

#Input YouTube API Key
api_key = ''

# YouTube API service
youtube = build('youtube', 'v3', developerKey=api_key)

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from operator import itemgetter
import os
comments_collect = pd.DataFrame()

def get_complete_channel_info(channel_id):
    
    try:
        channel_response = youtube.channels().list(
            part='snippet,contentDetails,statistics,topicDetails,status,brandingSettings',
            id=channel_id
        ).execute()

        if 'items' in channel_response and len(channel_response['items']) > 0:
            return channel_response['items'][0]
        else:
            return None

    except HttpError as e:
        print('An error occurred while fetching channel details:', e)
        return None


top_videos = []
def get_video_info(channel_id):
    video_data = []

    try:
        # Get uploads playlist ID of the channel
        channel_response = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        ).execute()
        print("channel_response",channel_response)
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # Retrieve video IDs from the uploads playlist
        playlist_response = youtube.playlistItems().list(
            part='snippet',
            playlistId=uploads_playlist_id,
            maxResults=20000
        ).execute()
        print(playlist_response)
        while playlist_response:
            for item in playlist_response['items']:
                video_id = item['snippet']['resourceId']['videoId']
                video_data.append({'id': video_id})

            if 'nextPageToken' in playlist_response:
                playlist_response = youtube.playlistItems().list(
                    part='snippet',
                    playlistId=uploads_playlist_id,
                    maxResults=20000,
                    pageToken=playlist_response['nextPageToken']
                ).execute()
            else:
                break

        video_ids = [item['id'] for item in video_data]

        video_info_list = []

        # Fetch details for each video
        for video_id in video_ids:
            video_info = youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            ).execute()['items'][0]                        
            

            video_entry = {
                'channel_id': channel_id,
                'video_id': video_id,
                'channelTitle': video_info['snippet']['channelTitle'],
                'title': video_info['snippet']['title'],
                'description': video_info['snippet']['description'],
                'tags': video_info['snippet'].get('tags', []),
                'category': video_info['snippet'].get('categoryId', ''),
                'duration': video_info['contentDetails']['duration'],
                'dimension': video_info['contentDetails']['dimension'],
                'definition': video_info['contentDetails']['definition'],
                'publish_time': video_info['snippet']['publishedAt'],
                'defaultAudioLanguage': video_info.get('defaultAudioLanguage', 'Not Available'),
                'views': int(video_info['statistics'].get('viewCount', 0)),
                'likes': int(video_info['statistics'].get('likeCount', 0)),
                'dislikes': int(video_info['statistics'].get('dislikeCount', 0)),
                'comments': int(video_info['statistics'].get('commentCount', 0)),
                'shares': int(video_info['statistics'].get('shareCount', 0)),
                'favorites': int(video_info['statistics'].get('favoriteCount', 0)),
            }
            video_info_list.append(video_entry)

            if video_id:
            # Create an empty list to store comments
                comments_list = []

                # Loop to retrieve all comments using pagination
                nextPageToken = None
                total_comments = 0

                while True:
                    # Call the API to retrieve comments for the video
                    comments = youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        textFormat='plainText',
                        maxResults=10000,  # Maximum number of comments per page
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
                        break  # Break the loop if no more pages

                    print(f"Collected {total_comments} comments so far...")

                print(f"Total comments collected: {total_comments}")

                # Create a DataFrame from the list of comments
                comments_df = pd.DataFrame(comments_list)
                comments_df['video_id'] = video_id               
                

            # Sort top_videos based on likes and views
        video_info_list.sort(key=itemgetter('likes', 'views'), reverse=True)     

        # Create a DataFrame for video information
        video_df = pd.DataFrame(video_info_list)

        comments_collect = pd.concat([comments_collect,comments_df], ignore_index=True) 
            

        return video_df, comments_collect_df

    except HttpError as e:
        print('An error occurred:', e)
        return None

def get_channel_info(channel_id):
    try:
        # Get complete channel information
        channel_json = get_complete_channel_info(channel_id)
        print("in channel", channel_id, channel_json)
        # Ensure there are items in the channel_json
        if 'snippet' in channel_json:
            # Extract channel information
            topicCategories = channel_json['topicDetails']['topicCategories']
            
            channel_info = {
                'channel_id': channel_id,
                'publishedAt': channel_json['snippet']['publishedAt'],
                'country': channel_json['snippet'].get('country', 'Not Available'),
                'viewCount': int(channel_json['statistics']['viewCount']),
                'subscriberCount': int(channel_json['statistics']['subscriberCount']),
                'videoCount': int(channel_json['statistics']['videoCount']),
                'topicCategories' : ', '.join(topicCategories),                
                'madeForKids': channel_json['status'].get('madeForKids', 'Not Available')
            }
            channel_df = pd.DataFrame([channel_info])  # Ensure it's a list of dict
            return channel_df
        else:
            return None  # or handle when 'items' list is empty or absent

    except Exception as e:
        print('An error occurred while fetching channel info:', e)
        return None


video_df, comments_df = get_video_info(channel_id)
channel_df = get_channel_info(channel_id)

video_df.to_csv('C:\\Users\\User\\OneDrive\\Documents\\Assignments\\Project\\Phase II\\data_collection\\video_df.csv')
channel_df.to_csv('C:\\Users\\User\\OneDrive\\Documents\\Assignments\\Project\\Phase II\\data_collection\\channel_df.csv')

# Iterate through each video and collect the comments

import pandas as pd
from googleapiclient.errors import HttpError

# List to store video IDs with disabled comments
disabled_comments_video_ids = []

comments_collect = pd.DataFrame()
c = 0
# Values to exclude
exclude_values = ['R9ZgvbxwS6o', 'UV8_2fblH7w','Oe0dWTU5VAk','A74HIuHd3k4','ei7BToGNH1M','J9B7CXCxWB0']

# Filter the DataFrame
filtered_video_ids = [video_id for video_id in missing_video_ids if video_id not in exclude_values]
for video_id in filtered_video_ids:
    #print("in")
    if video_id:
        # Create an empty list to store comments
        comments_list = []
        print(video_id)

        # Loop to retrieve all comments using pagination
        nextPageToken = None
        total_comments = 0

        try:
            while True:
                # Call the API to retrieve comments for the video
                comments = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    textFormat='plainText',
                    maxResults=10000,  # Maximum number of comments per page
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
                    break  # Break the loop if no more pages

        except HttpError as e:
            if e.resp.status == 403:  # 403 Forbidden - Comments are disabled
                disabled_comments_video_ids.append(video_id)
                continue  # Move to the next video ID

        #print(f"Total comments collected: {total_comments}")

        # Create a DataFrame from the list of comments
        comments_df = pd.DataFrame(comments_list)
        comments_df['video_id'] = video_id
        c = c + 1
        print(c)

        comments_collect = pd.concat([comments_collect,comments_df], ignore_index=True)


comments_collect_df = pd.DataFrame(comments_collect)
comments_collect_df = comments_collect_df[['video_id','Comment','Likes','Replies']]         
comments_collect_df.to_csv('C:\\Users\\User\\OneDrive\\Documents\\Assignments\\Project\\Phase II\\data_collection\\comments.csv')


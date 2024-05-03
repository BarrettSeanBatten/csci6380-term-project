# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:22:35 2024

@author: Barrett S. Batten
"""
import os
import json
# interact with the Twitter API via Tweepy
import tweepy
import pandas as pd
import time
# Import the csv module for access to QUOTE_NONNUMERIC
import csv  

# Authenticate Tweepy object to interact with Twitter API v2
# Collect bearer token stored in .json file
print("Current Working Directory:", os.getcwd())
with open("twitter_keys.json") as infile:
    json_obj = json.load(infile)
    token = json_obj["bearer_token"]
    # pip install tweepy

print(token)
client = tweepy.Client(bearer_token=token)

# File path for the CSV
file_path = 'tweets_2024_election_raw.csv'

# Define the search query with explicit candidate names and keywords
# For context, -has:links represents negated has links, filter tweets with links and exclude them
# More context, -is:retweet filters out retweets, we want original posts
query = (
    '('
    '"Donald J Trump" OR "Trump" OR "Joe Biden" OR "Biden" OR "RFK Jr" OR "Kennedy" OR '
    '"2024 Presidential Election" OR "November 5th" OR "@BidenHQ" OR "#MAGA" OR '
    '"#MakeAmericaGreatAgain" OR "#BidenHarris2024" OR "@TeamTrump" OR "#LetsGoBrandon" OR '
    '"#KennedyShanahan24"'
    ') '
    '-is:retweet -is:reply -has:links lang:en'
)
print("Length of query: " + str(len(query)))
# Specify the fields you want from the tweet and user
tweet_fields = ['author_id', 'created_at', 'text', 'lang']
user_fields = ['username']

# Variables to track tweet count and requests
total_tweets_collected = 0
max_tweets = 6504
max_requests_per_15_min = 60  # Adjust based on your rate limit understanding

# Continuously fetch tweets and handle pagination
next_token = None

while total_tweets_collected < max_tweets:
    try:
        # Fetch tweets using pagination and include user data
        response = client.search_recent_tweets(
            query=query,
            max_results=100,
            tweet_fields=tweet_fields,
            expansions='author_id',
            user_fields=user_fields,
            next_token=next_token
        )
        tweets = response.data
        users = {user.id: user.username for user in response.includes['users']}
        next_token = response.meta['next_token'] if 'next_token' in response.meta else None

        # Process tweets
        new_data = [
            [
                tweet.author_id,
                users.get(tweet.author_id, 'Unknown'),  # Fetch username using author_id
                tweet.created_at,
                tweet.text
                # tweet.geo['place_id'] if tweet.geo and 'place_id' in tweet.geo else None
            ]
            for tweet in tweets
        ]
        new_tweets_df = pd.DataFrame(new_data, columns=['User ID', 'Username', 'Date Created', 'Tweet'])
        # Convert user IDs to string to prevent scientific notation issues
        new_tweets_df['User ID'] = new_tweets_df['User ID'].astype(str)

        # Append to CSV file
        if os.path.exists(file_path):
            new_tweets_df.to_csv(file_path, mode='a', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
        else:
            new_tweets_df.to_csv(file_path, index=False)
        print(f"Appended {len(new_data)} new tweets to '{file_path}'. Total tweets collected: {total_tweets_collected + len(new_data)}")
        total_tweets_collected += len(new_data)

        # Manage rate limit
        if total_tweets_collected >= max_tweets or not next_token:
            print("Reached tweet collection goal or no more data available.")
            break
        time.sleep(15)  # Sleep to manage rate limit

    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(900)  # Sleep for 15 minutes if an error occurs
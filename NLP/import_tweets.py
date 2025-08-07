from twython import Twython
import pandas as pd

APP_KEY = 'YOUR_APP_KEY'
APP_SECRET = 'YOUR_APP_SECRET'
OAUTH_TOKEN = 'YOUR_OAUTH_TOKEN'
OAUTH_TOKEN_SECRET = 'YOUR_OAUTH_SECRET'

twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
justdoit_replies = []

for i in range(51):
    try:
        temp = twitter.search(q='#justdoit -filter:retweets',
                              lang='en',
                              result_type='recent',
                              count=100,
                              trim_user=False,
                              include_entities=True,
                              max_id=None if i == 0 else [x['id'] for x in justdoit_replies[-1]['statuses']][-1] - 1,
                              tweet_mode='extended')
        justdoit_replies.append(temp)
    except Exception as e:
        print(e)
        break
        
tweets_users_df = pd.DataFrame()
for tweets in justdoit_replies:
    tweets_temp = pd.DataFrame(tweets['statuses'])
    tweets_temp.columns = ['tweet_' + t for t in tweets_temp.columns]
    users_temp = pd.DataFrame([x['user'] for x in tweets['statuses']])
    users_temp.columns = ['user_' + u for u in users_temp.columns] 

    tweets_users_temp = pd.concat([tweets_temp,users_temp], axis=1)
    tweets_users_df = tweets_users_df.append(tweets_users_temp)

all([tweet_id == user_id for tweet_id, user_id in 
     zip([x['id'] for x in tweets_users_df['tweet_user']],tweets_users_df['user_id'])])

tweets_users_df.to_csv('justdoit_tweets_2018_09_07.csv', index=False)
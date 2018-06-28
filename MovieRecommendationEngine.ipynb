{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [],
   "source": [
    "# We will be using a MovieLens dataset. This dataset contains 100004 ratings across 1682 movies for 943 users. All selected users had at least rated 20 movies. We are going to build a recommendation engine which will suggest movies for a user which he hasn't watched yet based on the movies which he has already rated. We will be using k-nearest neighbour algorithm which we will implement from scratch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "movie_file = 'Data/movies.csv'\n",
    "movie_data = pd.read_csv(movie_file, usecols = [0, 1], encoding = 'utf-8')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [],
   "source": [
    "rating_file = 'Data/rating.csv'\n",
    "rating_info = pd.read_csv(rating_file, usecols = [0,1,2] )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MovieID</th>\n",
       "      <th>MovieName</th>\n",
       "      <th>UserID</th>\n",
       "      <th>Rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>308</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>287</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>148</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>280</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "      <td>66</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   MovieID         MovieName  UserID  Rating\n",
       "0        1  Toy Story (1995)     308       4\n",
       "1        1  Toy Story (1995)     287       5\n",
       "2        1  Toy Story (1995)     148       4\n",
       "3        1  Toy Story (1995)     280       4\n",
       "4        1  Toy Story (1995)      66       3"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movie_info = pd.merge(movie_data, rating_info, left_on = 'MovieID', right_on = 'MovieID')\n",
    "movie_info.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "943\n",
      "1682\n"
     ]
    }
   ],
   "source": [
    "num_user = max(movie_info.UserID)\n",
    "print(num_user)\n",
    "num_movies = max(movie_info.MovieID)\n",
    "print(num_movies)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "405    737\n",
       "655    685\n",
       "13     636\n",
       "450    540\n",
       "276    518\n",
       "Name: UserID, dtype: int64"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# how many movies were rated by each user \n",
    "# value_counts : If True then the object returned will contain the relative frequencies of the unique values.\n",
    "movies_per_user = movie_info.UserID.value_counts()\n",
    "movies_per_user.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# how many number of users rated each movie\n",
    "users_per_movie = movie_info.MovieName.value_counts()\n",
    "users_per_movie.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to find top N favourite movies of a user\n",
    "def fav_movies(current_user, N):\n",
    "    fav_movies = pd.DataFrame.sort_values(movie_info[movie_info.UserID == current_user], ['Rating'], ascending = [0]) [:N]\n",
    "    return list(fav_movies.MovieName)\n",
    "\n",
    "print(fav_movies(176, 5))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's create a matrix that has the user ids on one axis and the movie title on another axis. \n",
    "# Each cell will then consist of the rating the user gave to that movie. \n",
    "\n",
    "                                # Lets build recommendation engine now\n",
    "\n",
    "# We will use a neighbour based collaborative filtering model.\n",
    "# The idea is to use k-nearest neighbour algorithm to find neighbours of a user\n",
    "# We will use their ratings to predict ratings of a movie not already rated by a current user.\n",
    "# We will represent movies watched by a user in a vector - the vector will have values for all the movies in our dataset. If a user hasn't rated a movie, it would be represented as NaN.\n",
    "\n",
    "\n",
    "\n",
    "user_movie_rating_matrix  = pd.pivot_table(movie_info, values = 'Rating', index=['UserID'], columns=['MovieID'])\n",
    "user_movie_rating_matrix.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Now, we will find the similarity between 2 users by using correlation\n",
    "from scipy.spatial.distance import correlation\n",
    "import numpy as np\n",
    "def similarity(user1, user2):\n",
    "    # normalizing user1 rating i.e mean rating of user1 for any movie\n",
    "    # nanmean will return mean of an array after ignore NaN values \n",
    "    user1 = np.array(user1) - np.nanmean(user1) \n",
    "    user2 = np.array(user2) - np.nanmean(user2)\n",
    "    \n",
    "    # finding the similarity between 2 users\n",
    "    # finding subset of movies rated by both the users\n",
    "    common_movie_ids = [i for i in range(len(user1)) if user1[i] > 0 and user2[i] > 0]\n",
    "    if(len(common_movie_ids) == 0):\n",
    "        return 0\n",
    "    else:\n",
    "        user1 = np.array([user1[i] for i in common_movie_ids])\n",
    "        user2 = np.array([user2[i] for i in common_movie_ids])\n",
    "        return correlation(user1, user2)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "# We will now use the similarity function to find the nearest neighbour of a current user\n",
    "# nearest_neighbour_ratings function will find the k nearest neighbours of the current user and\n",
    "# then use their ratings to predict the current users ratings for other unrated movies \n",
    "def nearest_neighbour_ratings(current_user, K):\n",
    "     # Creating an empty matrix whose row index is userId and the value\n",
    "    # will be the similarity of that user to the current user\n",
    "    similarity_matrix = pd.DataFrame(index = user_movie_rating_matrix.index, \n",
    "                                    columns = ['similarity'])\n",
    "    for i in user_movie_rating_matrix.index:\n",
    "        # finding the similarity between user i and the current user and add it to the similarity matrix\n",
    "        similarity_matrix.loc[i] = similarity(user_movie_rating_matrix.loc[current_user],\n",
    "                                             user_movie_rating_matrix.loc[i])\n",
    "        # Sorting the similarity matrix in descending order\n",
    "    similarity_matrix = pd.DataFrame.sort_values(similarity_matrix,\n",
    "                                                ['similarity'], ascending= [0])\n",
    "    # now we will pick the top k nearest neighbour\n",
    "    # neighbour_movie_ratings : ratings of movies of neighbors\n",
    "    # user_movie_rating_matrix : ratings of each user for every movie\n",
    "    # predicted_rating : Averge where rating is NaN\n",
    "    nearest_neighbours = similarity_matrix[:K]\n",
    "    neighbour_movie_ratings = user_movie_rating_matrix.loc[nearest_neighbours.index]\n",
    "     # This is empty dataframe placeholder for predicting the rating of current user using neighbour movie ratings\n",
    "    predicted_movie_rating = pd.DataFrame(index = user_movie_rating_matrix.columns, columns = ['rating'])\n",
    " # Iterating all movies for a current user\n",
    "    for i in user_movie_rating_matrix.columns:\n",
    "        # by default, make predicted rating as the average rating of the current user\n",
    "        predicted_rating = np.nanmean(user_movie_rating_matrix.loc[current_user])\n",
    "          # j is user , i is movie\n",
    "        for j in neighbour_movie_ratings.index:\n",
    "            # if user j has rated the ith movie\n",
    "            if(user_movie_rating_matrix.loc[j,i] > 0):# If there is some rating  # nearest_neighbours.loc[j, 'similarity']) / nearest_neighbours['similarity'].sum(): Finding Similarity score\n",
    "                predicted_rating += ((user_movie_rating_matrix.loc[j,i] - np.nanmean(user_movie_rating_matrix.loc[j])) *\n",
    "                                                    nearest_neighbours.loc[j, 'similarity']) / nearest_neighbours['similarity'].sum()\n",
    "\n",
    "        predicted_movie_rating.loc[i, 'rating'] = predicted_rating\n",
    "\n",
    "    return predicted_movie_rating\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predicting top N recommendations for a current user\n",
    "def top_n_recommendations(current_user, N):\n",
    "    predicted_movie_rating = nearest_neighbour_ratings(current_user, 10)\n",
    "    movies_already_watched = list(user_movie_rating_matrix.loc[current_user]\n",
    "                                  .loc[user_movie_rating_matrix.loc[current_user] > 0].index)\n",
    "    \n",
    "    predicted_movie_rating = predicted_movie_rating.drop(movies_already_watched)\n",
    "    \n",
    "    top_n_recommendations = pd.DataFrame.sort_values(predicted_movie_rating, ['rating'], ascending=[0])[:N]\n",
    "    \n",
    "    top_n_recommendation_titles = movie_data.loc[movie_data.MovieID.isin(top_n_recommendations.index)]\n",
    "\n",
    "    return list(top_n_recommendation_titles.MovieName)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\reshu\\Anaconda3\\lib\\site-packages\\scipy\\spatial\\distance.py:644: RuntimeWarning: invalid value encountered in double_scalars\n",
      "  dist = 1.0 - uv / np.sqrt(uu * vv)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "User's favorite movies are :  ['English Patient, The (1996)', \"Ulee's Gold (1997)\", 'Fly Away Home (1996)', 'Chasing Amy (1997)', 'Soul Food (1997)'] \n",
      "User's top recommendations are:  ['Star Wars (1977)', 'Shawshank Redemption, The (1994)', 'Godfather, The (1972)']\n"
     ]
    }
   ],
   "source": [
    "# finding out the recommendations for a user\n",
    "current_user = 140\n",
    "print(\"User's favorite movies are : \", fav_movies(current_user, 5),\n",
    "      \"\\nUser's top recommendations are: \", top_n_recommendations(current_user, 3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

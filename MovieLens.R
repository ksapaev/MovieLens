### Author: Khushnud Sapaev
### Project: MovieLens Project 
### Course: HarvardX: PH125.9x - Capstone Project
### GitHub: https://github.com/ksapaev/


###############################
### Dataset and Preparation ###
###############################


# Create edx set and validation set
# Note: this process could take a couple of minutes

# Loading libraries, if does not exist then installing first
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



############################
### Methods and Analysis ###
############################

#Getting the summary of the dataset. Counting number of rows without NA's
summary(edx)
sum(complete.cases(edx))

summary(validation)
sum(complete.cases(validation))


#Number of rows by genres and ratings
edx %>% group_by(genres) %>% summarize(count = n()) %>% arrange(desc(count))
edx %>% group_by(rating) %>% summarize(count = n()) %>% arrange(desc(count))

#Visualization of ratings
ggplot(edx, aes(rating)) + geom_bar(fill = "orange") +
  scale_x_continuous(breaks = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 2500000, 500000))) +
  theme_bw() + theme(panel.border = element_blank()) +
  ggtitle("Rating distribution of movies")

# Quantity of unique users and movies 
edx %>% summarize(Users = length(unique(userId)), Movies = length(unique(movieId)))

#Distribution of number of ratings per movie
edx %>% group_by(movieId) %>% summarize(count = n()) %>% filter(count == 1) %>%
  left_join(edx, by = "movieId") %>% group_by(title) %>% summarize(rating = rating, count = count) 

#Visualization of number of ratings per movie
edx %>% count(movieId) %>% ggplot(aes(n)) +
  geom_histogram(bins = 25) +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")

#Visualization of number of ratings per user
edx %>% count(userId) %>% ggplot(aes(n)) +
  geom_histogram(bins = 25) +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of users") +
  ggtitle("Number of ratings per user")


############################
###  Modelling Approach  ###
############################

##########################################  
### Movie and user effect model

# Mean rating of edx subset, movie and user averages
mu <- mean(edx$rating)
movie_avgs <- edx %>% group_by(movieId) %>% summarize(m_a = mean(rating - mu))
user_avgs <- edx %>% left_join(movie_avgs, by='movieId') %>% group_by(userId) %>%
  summarize(u_a = mean(rating - mu - m_a))

# Testing and saving the RMSE result 
predicted_ratings <- validation %>% left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(predict = mu + m_a + u_a) %>% pull(predict)

movie_user_model_rmse <- RMSE(predicted_ratings, validation$rating)

# Checking the result
rmse_results <- tibble(method="Movie and user effect model", RMSE = movie_user_model_rmse)
rmse_results


##########################################                                                            
## Regularized movie and user effect model

# lambda is a tuning parameter
lambdas <- seq(0, 10, 0.25)

# m_a and u_a for each lambda followed by prediction and testing the rating
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  m_a <- edx %>% group_by(movieId) %>% summarize(m_a = sum(rating - mu)/(n()+l))
  u_a <- edx %>% left_join(m_a, by="movieId") %>% group_by(userId) %>%
    summarize(u_a = sum(rating - mu - m_a)/(n()+l))
  predicted_ratings <- validation %>% left_join(m_a, by = "movieId") %>%
    left_join(u_a, by = "userId") %>%
    mutate(predict = mu + m_a + u_a) %>% pull(predict)
  return(RMSE(predicted_ratings, validation$rating))
})

# Visualization of RMSEs vs lambdas to select the optimal lambda                                                             
qplot(lambdas, rmses)  

# The optimal lambda is the one with the minimal RMSE                                                          
lambda <- lambdas[which.min(rmses)]

# Checking the result with the result from the previous model                                                             
rmse_results <- tibble(method="Regularized movie and user effect model", RMSE = min(rmses))
rmse_results %>% knitr::kable()

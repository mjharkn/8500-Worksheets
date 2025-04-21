# Word Vector Analysis of "The Woman Citizen" Magazine

library(wordVectors)
library(tidyverse)
library(tidytext)
library(ggplot2)
library(dplyr)

# My historical questions are: how does the way the The Woman Citizen magazine talk about voting/suffrage change over the course of three months (June, July, August)? AND How does the magazine talk about men vs women, and does this shift over the course of three months?

download.file("https://github.com/regan008/8510-TextAnalysisData/raw/423e2ec8c0f82b865cdfd09ef0a8ddead51d7292/TheWomanCitizen.zip", "TheWomanCitizen.zip")
unzip("TheWomanCitizen.zip")

wc <- readtext(paste(getwd(), "TheWomanCitizen/*.txt", sep = "/"))

wc.metadata <- read.csv("https://github.com/regan008/8510-TextAnalysisData/raw/423e2ec8c0f82b865cdfd09ef0a8ddead51d7292/WomanCitizenMetadata.csv")
# The metadata for the Woman Citizen magazine. I am making a metadata set so that I can train separate models based on month.

# Making sure file names are consistent
wc.metadata <- wc.metadata %>% 
    mutate(filename = paste0("WC_", filename))

wc <- wc %>%
    mutate(doc_id = paste("WC_", doc_id))

# Joining the metadata with text data
wc.whole <- full_join(wc.metadata, wc, by = c("filename" = "doc_id"))

# Some of the files had NA values for the month, but since I know the month it was published I manually mutated each incorrect entry to get rid of the NA values so that my model will be more complete.
wc.whole <- wc.whole %>%
    mutate(month = case_when(
        filename == "WC_ August10_1918.txt" ~ "August",
        filename == "WC_ August3_1918.txt" ~ "August",
        filename == "WC_ August31_1918.txt" ~ "August",
        filename == "WC_ July12_1918.txt" ~ "July",
        filename == "WC_ July20_1918.txt" ~ "July",
        filename == "WC_ July27_1918.txt" ~ "July",
        filename == "WC_ July9_1918.txt" ~ "July",
        filename == "WC_ June1_1918.txt" ~ "June",
        filename == "WC_ June15_1918.txt" ~ "June",
        filename == "WC_ June22_1918.txt" ~ "June",
        filename == "WC_ June8_1918.txt" ~ "June",
        TRUE ~ month
    ))

# Saving preprocessed text for each month and creating a directory for them
if (!dir.exists("monthly_texts")) dir.create("monthly_texts")

# Combining all text for each month
wc.grouped <- wc.whole %>%
  group_by(month) %>%
  summarize(text = paste(text, collapse = " "))

# Writing and saving each month's text to a separate file
for (i in 1:nrow(wc.grouped)) {
  file_path <- paste0("monthly_texts/wc_", wc.grouped$month[i], ".txt")
  write_lines(wc.grouped$text[i], file_path)
}

# Training a  model for each month
models <- list()  # Storing models in a list; I can call separately by month
for (month in unique(wc.whole$month)) {
  text_file <- paste0("monthly_texts/wc_", month, ".txt")
  model_file <- paste0("monthly_texts/wc_", month, ".bin")
  
  if (!file.exists(model_file)) {
    models[[month]] <- train_word2vec(
      text_file, model_file,
      vectors = 150, threads = 1, window = 12, iter = 5, negative_samples = 0
    )
  } else {
    models[[month]] <- read.vectors(model_file)
  }
}

# Analyzing models for specific months
models[["August"]] %>% closest_to("vote")
models[["July"]] %>% closest_to("vote")  
models[["June"]] %>% closest_to("vote")  

## Visualizing the word vectors for each month

# For June
june.vote <- models[["June"]][[c("vote", "women"), average = FALSE]]
    june.vote.vector <- models[["June"]][1:300, ] %>% cosineSimilarity(june.vote)
    june.vote.vector[
         rank(-june.vote.vector[, 1]) < 20 |
            rank(-june.vote.vector[, 2]) < 20,
    ]
plot(june.vote.vector, type = "n")
text(june.vote.vector, labels = rownames(june.vote.vector))

# For July
july.vote <- models[["July"]][[c("vote", "women"), average = FALSE]]
    july.vote.vector <- models[["July"]][1:300, ] %>% cosineSimilarity(july.vote)
    july.vote.vector[
         rank(-july.vote.vector[, 1]) < 20 |
            rank(-july.vote.vector[, 2]) < 20,
    ]
plot(july.vote.vector, type = "n")
text(july.vote.vector, labels = rownames(july.vote.vector))

# For August
august.vote <- models[["August"]][[c("vote", "women"), average = FALSE]]
    august.vote.vector <- models[["August"]][1:300, ] %>% cosineSimilarity(august.vote) # I lowered the vocab size because this corpus is small and it was also unreadable when it was in the thousands.
    august.vote.vector[
         rank(-august.vote.vector[, 1]) < 20 |
            rank(-august.vote.vector[, 2]) < 20,
    ]
plot(august.vote.vector, type = "n")
text(august.vote.vector, labels = rownames(august.vote.vector))


## Getting most common words for a group of terms; I am using terms associated with women and suffrage. I am also interested in comparing words associated with physical (protest) and written (write) action just to compare. 

# For June
june.suffrage <- models[["June"]][[c("vote", "women", "protest", "write"), average = FALSE]]
common_similiarities_vote <- models[["June"]][1:300, ] %>% cosineSimilarity(june.suffrage)
common_similiarities_vote[20:30, ]

high_similarities_to_women_vote <- common_similiarities_vote[rank(-apply(common_similiarities_vote, 1, max)) < 75, ]
high_similarities_to_women_vote %>%
    prcomp() %>%
    biplot(main = "Projection of women and suffrage for June 1918")

# For July
july.suffrage <- models[["July"]][[c("vote", "women", "protest", "write"), average = FALSE]]
common_similiarities_vote <- models[["July"]][1:300, ] %>% cosineSimilarity(july.suffrage)
common_similiarities_vote[20:30, ]

high_similarities_to_women_vote <- common_similiarities_vote[rank(-apply(common_similiarities_vote, 1, max)) < 75, ]
high_similarities_to_women_vote %>%
    prcomp() %>%
    biplot(main = "Projection of women and suffrage for July 1918")


# For August
aug.suffrage <- models[["August"]][[c("vote", "women", "protest", "write"), average = FALSE]]
common_similiarities_vote <- models[["August"]][1:300, ] %>% cosineSimilarity(aug.suffrage)
common_similiarities_vote[20:30, ]

high_similarities_to_women_vote <- common_similiarities_vote[rank(-apply(common_similiarities_vote, 1, max)) < 75, ]
high_similarities_to_women_vote %>%
    prcomp() %>%
    biplot(main = "Projection of women and suffrage for August 1918")


## I am interested because women and men always seem to be consistently close throughout all the months. I am interested now in seeing how this publication talks about women vs men and whether or not there is any variation across months, so I am going to find the most common words associated with each.

# For June
june.word.scores <- data.frame(word = rownames(models[["June"]]))

june.word.scores$gender.score <- models[["June"]] %>%
  cosineSimilarity(models[["June"]][[c("feminine", "feminity", "woman", "women")]] -
                   models[["June"]][[c("masculine", "masculinity", "men", "man")]]) %>%
  as.vector()

ggplot(june.word.scores %>% filter(abs(gender.score) > .33)) +
    geom_bar(aes(y = gender.score, x = reorder(word, gender.score), fill = gender.score < 0), stat = "identity") +
    coord_flip() +
    scale_fill_discrete("Indicative of gender", labels = c("Feminine", "masculine")) +
    labs(title = "Gender Binary Words for June 1918")

# For July
july.word.scores <- data.frame(word = rownames(models[["July"]]))

july.word.scores$gender.score <- models[["July"]] %>%
  cosineSimilarity(models[["July"]][[c("feminine", "feminity", "woman", "women")]] -
                   models[["July"]][[c("masculine", "masculinity", "men", "man")]]) %>%
  as.vector()

ggplot(july.word.scores %>% filter(abs(gender.score) > .33)) +
    geom_bar(aes(y = gender.score, x = reorder(word, gender.score), fill = gender.score < 0), stat = "identity") +
    coord_flip() +
    scale_fill_discrete("Indicative of gender", labels = c("Feminine", "masculine")) +
    labs(title = "Gender Binary Words for July 1918")

# For August
aug.word.scores <- data.frame(word = rownames(models[["August"]]))

aug.word.scores$gender.score <- models[["August"]] %>%
  cosineSimilarity(models[["August"]][[c("feminine", "feminity", "woman", "women")]] -
                   models[["August"]][[c("masculine", "masculinity", "men", "man")]]) %>%
  as.vector()

ggplot(aug.word.scores %>% filter(abs(gender.score) > .33)) +
    geom_bar(aes(y = gender.score, x = reorder(word, gender.score), fill = gender.score < 0), stat = "identity") +
    coord_flip() +
    scale_fill_discrete("Indicative of gender", labels = c("Feminine", "masculine")) +
    labs(title = "Gender Binary Words for July 1918")


## Clustering

# For June
set.seed(10)
centers <- 150
june.clustering <- kmeans(models[["June"]], centers = centers, iter.max = 40)

sapply(sample(1:centers, 10), function(n) {
    names(june.clustering$cluster[june.clustering$cluster == n][1:10])
})

# For July
set.seed(10)
centers <- 150
july.clustering <- kmeans(models[["July"]], centers = centers, iter.max = 40)

sapply(sample(1:centers, 10), function(n) {
    names(july.clustering$cluster[july.clustering$cluster == n][1:10])
})

# For August
set.seed(10)
centers <- 150
aug.clustering <- kmeans(models[["July"]], centers = centers, iter.max = 40)

sapply(sample(1:centers, 10), function(n) {
    names(aug.clustering$cluster[aug.clustering$cluster == n][1:10])
})

## Dendogram

# For June
june.suffrage.dendro <- c("write", "protest", "vote", "suffrage")
term_set <- lapply(
    june.suffrage.dendro,
    function(june.suffrage.dendro) {
        nearest_words <- models[["June"]] %>% closest_to(models[["June"]][[june.suffrage.dendro]], 20)
        nearest_words$word
    }
) %>% unlist()
subset <- models[["June"]][[term_set, average = F]]
subset %>%
    cosineDist(subset) %>%
    as.dist() %>%
    hclust() %>%
    plot()

# For July
july.suffrage.dendro <- c("write", "protest", "vote", "suffrage")
term_set <- lapply(
    july.suffrage.dendro,
    function(july.suffrage.dendro) {
        nearest_words <- models[["July"]] %>% closest_to(models[["July"]][[july.suffrage.dendro]], 20)
        nearest_words$word
    }
) %>% unlist()
subset <- models[["July"]][[term_set, average = F]]
subset %>%
    cosineDist(subset) %>%
    as.dist() %>%
    hclust() %>%
    plot()

# For August
aug.suffrage.dendro <- c("write", "protest", "vote", "suffrage")
term_set <- lapply(
    aug.suffrage.dendro,
    function(aug.suffrage.dendro) {
        nearest_words <- models[["August"]] %>% closest_to(models[["August"]][[aug.suffrage.dendro]], 20)
        nearest_words$word
    }
) %>% unlist()
subset <- models[["August"]][[term_set, average = F]]
subset %>%
    cosineDist(subset) %>%
    as.dist() %>%
    hclust() %>%
    plot()

### Summary of Findings

# Before performing a word vector analysis on this corpus, I knew from previous text analysis and topic modeling that it focused on the women's suffrage movement. I was therefore interseted in exploring how this publication went about discussing voting and women earning the right to vote. I also knew that it was a fairly small corpus and divided into the three summer months (June, July, and August), so I was interested in doing a filtered analysis of each of these months to see if there were any variations throughout that would reflect historical change.

# 
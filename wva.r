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

# For these first plots, there is a notable shift in focus in July and August towards what I believe to be the war effort. 
# I am making this observation based on words like “men”, “support”, “national”, “war”, “help”, “service”, “home”, and “country” being clustered together in July and August. 
# This shift makes sense, since these were the months leading up to the end of WWI. 

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

# For this plot, one thing overall I was interested in was seeing was seeing what type of action/activism The Woman Citizen encouraged. I chose to include protest and write. 
# These plots showed me that across all of the months, there were more words clustered around protest than write. 
# This tells me that this publication prioritized promoting more direct and physical activism like protesting and marching over written activism. 
# Another observation is that there is a continued focus on supporting men and the war effort that is present throughout all months. 
# Words like “world”, “war”, “power”, “men”, “support”, “service”, “country”, and “children” give a good idea of the rhetoric being used; there is a focus on women supporting the war effort, the country, and the family.

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
    labs(title = "Gender Binary Words for August 1918")

# Since men seem to be consistently appearing in this publication, I am interested in seeing how the text talks about men. 
# Something that stands out right away is how many more words associated with men entries from June had in comparison to July and August. 
# Furthermore, although there is a notable shift in July and August towards feminine words, these words (i.e. “military”, “vet”, “hope”, “father”, “serve”, etc.) are still largely focused around the war effort, they remain associated with women. 
# This demonstrates that despite the rhetoric of The Woman Citizen being dedicated in the months of July and August to a “masculine” cause, the audience was still women. 
# It is still clear that the war was influencing the rhetoric of this publication at the time.
# Another interesting thing to note is mention of Woodrow Wilson in July. 
# This could either reflect simply discussion of him due to the war, or, it could be due to his support in 1918 of the women’s suffrage amendment. 
# Either way, it shows the president’s importance to the publication.

## Dendograms
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

# I found the dendrograms useful for providing insight into the contextuality of the use of certain key words. 
# We see voting being connected to the military, writing being connected to words like “send”, “talk”, “him”, (I am guessing this is referring to sending letters). 
# June and July especially seem to be reflecting wartime rhetoric. 
# Also of note here is that words like “hospitals”, “preparing”, “country”, and “law” being clustered near protest and vote could reflect the onset of the Spanish flu pandemic which was beginning to spread rapidly during this time.


### Summary of Findings

# Before performing a word vector analysis on this corpus, I knew from previous text analysis and topic modeling that it focused on the women's suffrage movement. 
# I was therefore interseted in exploring how this publication went about discussing voting and women earning the right to vote. I also knew that it was a fairly small corpus and divided into the three summer months (June, July, and August), so I was interested in doing a filtered analysis of each of these months to see if there were any variations throughout that would reflect historical change.

# Overall, the main way I noticed that The Woman Citizen talks about voting related to women is that it utilizes wartime rhetoric a fair amount, particularly in June and July. 
# This demonstrates the importance of WWI to the women’s suffragist movement, and how it was likely common to employ tactics that promote women’s suffrage as being good for the benefit of the country and American soldiers (many of whom were fathers and husbands of suffragists).
# Although there was a significantly larger amount of masculine associated words in June entries, The Woman Citizen overall is clearly directed towards a female audience. 
# I am making this observation based on the fact that in July and August, feminine associated words dominated and reflected activism and political action (i.e. “federal”, “resolutions”, “convention”, “primary”, “candidates”, “petition”, “passed”, and “campaign”). 
# This tells me that this publication encouraged its women readers to have an active role in the suffragist movement. 

# Furthermore, the way that The Woman Citizen talks about men seems largely attached to women. 
# This is demonstrated in June by looking at the masculine associated words, some of which include “help”, “hope”, “father”, “pledge”, “vet”, “husband”, and “salary”, highlight the connected of the man to the woman, particularly relating to their potential role in the war. 
# This further supports the overarching observation that wartime/military rhetoric was a popular tactic for publications serving the women’s suffrage movement. 

# The main shift over the course of the three months of this publication seems to be that there was a greater emphasis in June and July on warfare rhetoric, 
# with more emphasis on women’s protest and political action in August. 

# After performing a word vector analysis on this corpusl, I am left with a greater understanding of the rhetoric used in this publication.
# It is clear that The Woman Citizen was dedicated to promoting women's suffrage through supporting the war effort. 
# This is demonstrated through the use of military and wartime rhetoric, as well as the focus on women’s political action and activism associated with the war.
# Furthermore, it is clear by observing the strong push of wartime rhetoric in June and July means reflects the historical context of the end of the war giving the publication a new focus and strategy.
# It is therefore overall clear that this publication is potentially reflective of the braoder trend of the women's suffrage movement being influenced heavily by the war.


library(ScottKnott)
library(ggplot2)
library(dplyr)
library(stringr)

# Load data
df <- read.csv('../rq2_results/1_fullturn/2_analyzing/0_8_turn_counts_final_ids.csv')

# Filter only topics 0–9 and valid data
df_top <- df %>%
  filter(!is.na(turns), final_cluster_id %in% 0:9)

# Convert to factor for SK
df_top$topic_factor <- factor(df_top$final_cluster_id)

cat("✅ Sample size per topic:\n")
print(table(df_top$topic_factor))

# Run Scott-Knott
sk_result <- tryCatch({
  SK(turns ~ topic_factor, data = df_top, which = "topic_factor")
}, error = function(e) {
  cat("❌ Error in SK():\n")
  print(e)
  quit(status = 1)
})

# Extract SK group letters
sk_summary <- as.data.frame(summary(sk_result))
group_cols <- grep("^G[0-9]+$", colnames(sk_summary), value = TRUE)
group_letters <- apply(sk_summary[, group_cols], 1, function(row) {
  non_empty <- which(row != "")
  if (length(non_empty) == 1) return(row[non_empty]) else return(NA)
})

group_map <- data.frame(
  topic_factor = rownames(sk_summary),
  SK_group = group_letters,
  stringsAsFactors = FALSE
)

# Merge back
df_top <- merge(df_top, group_map, by = "topic_factor")

# Create clean label: Topic N only (no SK info)
df_top$topic_id <- as.numeric(as.character(df_top$topic_factor))
df_top$display_label <- paste0("Topic ", df_top$topic_id + 1)

# Ensure SK_group is an ordered factor (a–e)
df_top$SK_group <- factor(df_top$SK_group, levels = sort(unique(df_top$SK_group)))

# Order topics within SK group
ordered_labels <- df_top %>%
  distinct(display_label, SK_group, topic_id) %>%
  arrange(SK_group, topic_id) %>%
  pull(display_label)

df_top$display_label <- factor(df_top$display_label, levels = ordered_labels)

# Define SK group colours
sk_group_levels <- levels(df_top$SK_group)
group_colours <- RColorBrewer::brewer.pal(n = length(sk_group_levels), name = "Set2")
names(group_colours) <- sk_group_levels

# Map fill colour by SK group instead of display_label
gg <- ggplot(df_top, aes(x = display_label, y = turns, fill = SK_group)) +
  geom_violin(trim = FALSE, alpha = 0.5) +
  geom_boxplot(width = 0.1, outlier.shape = NA, fill = "white") +
  scale_fill_manual(values = group_colours) +
  scale_y_continuous(trans = "log10", breaks = c(1, 2, 5, 10, 20, 50, 100)) +
  labs(x = NULL, y = "Turn Count per Conversation") +
  facet_grid(. ~ SK_group, scales = "free_x", space = "free_x") +
  theme_minimal() +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "grey90", colour = "black"),
    strip.text = element_text(size = 18, face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 18, face = "bold"),
    axis.text.y = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 16, face = "bold"),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white")
  )

# Save plot
ggsave(
  "../../rq2_results/1_fullturn/2_analyzing/RQ2_TurnsInTopicsFullturn.pdf",
  gg, width = 12, height = 4, dpi = 300, bg = "white"
)

cat("✅ Saved with Scott-Knott group labels as upper-level facets\n")
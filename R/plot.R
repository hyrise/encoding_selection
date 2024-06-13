library(dplyr)
library(ggplot2)
library(ggrepel)


scale_factor = Sys.getenv("SCALE_FACTOR")
run_name = Sys.getenv("CALIBRATION_RUN")  # currently not used.
hyrise_core_count = Sys.getenv("HYRISE_CORE_COUNT")
hyrise_client_count = Sys.getenv("HYRISE_CLIENT_COUNT")
comparison_core_count = Sys.getenv("HYRISE_CORE_COUNT")
comparison_client_count = Sys.getenv("HYRISE_CLIENT_COUNT")

results_dir = paste0("results_to_plot")


####
#### BUDGET-CONSTRAINED HYRISE
####

suffix <- paste0(hyrise_client_count, "_clients_and_", hyrise_core_count, "_cores.csv")
hyrise_runtimes <- read.csv(paste0(results_dir, "/LPCompressionSelection/runtimes__", suffix))
hyrise_runtimes_sum <- hyrise_runtimes %>% filter(ITEM_NAME == "All queries") %>% group_by(MODEL, BUDGET) %>% summarize(RUN_RUNTIME_MS = median(RUNTIME_MS), .groups="keep")
hyrise_sizes <- read.csv(paste0(results_dir, "/LPCompressionSelection/sizes__", suffix))
hyrise <- inner_join(hyrise_runtimes_sum, hyrise_sizes)

actual_execution_count <- nrow(hyrise_runtimes) / nrow(hyrise_runtimes %>% distinct(MODEL, BUDGET, ITEM_NAME))
hyrise$runs_per_hour <- 3600000 / hyrise$RUN_RUNTIME_MS
hyrise$size_gb <- hyrise$SIZE_IN_BYTES / 1000 / 1000 / 1000
hyrise_lp <- hyrise %>% filter(MODEL == "LPCompressionSelection")
hyrise_lp$DATABASE_SYSTEM <- "Budget-Constrained Hyrise"
hyrise_lp$is_geom_line <- TRUE


####
#### REST
####

monet_runtimes <- read.csv(paste0(results_dir, "/database_comparison__TPC-H__monetdb.csv"))
hyrise_runtimes <- read.csv(paste0(results_dir, "/database_comparison__TPC-H__hyrise.csv"))
duckdb_runtimes <- read.csv(paste0(results_dir, "/database_comparison__TPC-H__duckdb.csv"))
hyrise_master_runtimes <- read.csv(paste0(results_dir, "/database_comparison__TPC-H__hyrise_master.csv"))
hyrise_master_runtimes$DATABASE_SYSTEM = "hyrise_master"

monet_size <- read.csv(paste0(results_dir, "/size_monetdb__SF", scale_factor, ".csv"))
hyrise_size <- read.csv(paste0(results_dir, "/size_hyrise__SF", scale_factor, ".csv"))
duckdb_size <- read.csv(paste0(results_dir, "/size_duckdb__SF", scale_factor, ".csv"))
hyrise_master_size <- read.csv(paste0(results_dir, "/size_hyrise_master__SF", scale_factor, ".csv"))
hyrise_master_size$DATABASE_SYSTEM = "hyrise_master"

runtimes <- rbind(monet_runtimes, hyrise_runtimes, duckdb_runtimes)
sizes <- rbind(monet_size, hyrise_size, duckdb_size)

runtimes_q_agg <- runtimes %>% group_by(DATABASE_SYSTEM, ITEM_NAME) %>% summarize(median_runtime = mean(RUNTIME_MS), .groups="keep")
runtimes_db_agg <- runtimes_q_agg %>% group_by(DATABASE_SYSTEM) %>% summarize(cumulative_runtime = sum(median_runtime), .groups="keep")

joined <- inner_join(runtimes_db_agg, sizes)
joined$size_gb <- joined$SIZE_IN_BYTES / 1000 / 1000 / 1000
joined$runtime_s <- joined$cumulative_runtime / 1000
joined$runs_per_hour <- 3600 / joined$runtime_s
joined$is_geom_line <- FALSE

joined <- rbind(joined, hyrise_lp)

# Copy of last budget point for easier labeling
first_lp <- hyrise_lp %>% arrange(desc(SIZE_IN_BYTES)) %>% tail(1)
first_lp$is_geom_line <- FALSE
joined <- rbind(joined, first_lp)

# Renamings
joined$DATABASE_SYSTEM[which(joined$DATABASE_SYSTEM == "duckdb")] <- "DuckDB"
joined$DATABASE_SYSTEM[which(joined$DATABASE_SYSTEM == "monetdb")] <- "MonetDB"
joined$DATABASE_SYSTEM[which(joined$DATABASE_SYSTEM == "hyrise")] <- "Default Hyrise"
joined$DATABASE_SYSTEM[which(joined$DATABASE_SYSTEM == "hyrise_master")] <- "Hyrise Master"

max_size <- max(joined$size_gb)
max_throughput <- max(joined$runs_per_hour)

g <- ggplot(joined, aes(x=size_gb, y=runs_per_hour, group=DATABASE_SYSTEM, fill=DATABASE_SYSTEM,
                        linetype=DATABASE_SYSTEM, shape=DATABASE_SYSTEM, color=DATABASE_SYSTEM)) +
  geom_line(data=joined[joined$is_geom_line == TRUE, ], size=0.8) +
  geom_point(data=joined[joined$is_geom_line == TRUE, ], size=1.5) +
  geom_point(data=joined[joined$is_geom_line == FALSE, ], size=2.0) +
  xlab(expression(paste("Memory consumption (full TPC-H data set) [GB]"))) +
  ylab(expression(paste("TPC-H throughput [runs per hour/client]"))) +
  labs(title = paste("TPC-H Results - Scale Factor", scale_factor),
       subtitle = "Not meant to be a fair end-to-end comparison. ",
       caption = paste0("Note: Not meant to be a fair end-to-end comparison. GitHub actions might\nrun on different",
                        " VMs without any performance or system guarantees\n(Hyrise with ", hyrise_core_count,
                        " cores and ", hyrise_client_count, " clients; others with ", comparison_core_count,
                        " cores and ", comparison_client_count, " clients).")) +
  guides(col = guide_legend(nrow = 2)) +
  scale_x_continuous(lim = c(0, 1.05*max_size), labels=function(x) format(x, big.mark = ",", scientific = FALSE)) +
  scale_y_continuous(lim = c(0, 1.05*max_throughput), labels=function(y) format(y, big.mark = ",", scientific = FALSE)) +
  scale_colour_brewer(palette="Set1") +
  guides(fill=guide_legend(nrow=2, byrow=FALSE)) +
  theme(legend.position="none") +
  geom_text_repel(
    data=joined[joined$is_geom_line == FALSE, ],
    aes(label = DATABASE_SYSTEM, group=DATABASE_SYSTEM),
    segment.size  = 0.2,
    force = 0.5,
  )

ggsave("db_comparison.pdf", g, width=7, height=5)
